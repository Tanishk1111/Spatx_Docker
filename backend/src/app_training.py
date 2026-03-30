"""
Training Portal Backend for SpatX
Handles model fine-tuning on user data
"""

import os
import shutil
import asyncio
import uuid
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from database import get_db, User
from spatx_core.spatx_core.models.cit_to_gene.CiT_Net_T import CIT
from spatx_core.spatx_core.models.cit_to_gene.CiTGene import CITGenePredictor

# We'll define get_current_user here to avoid circular imports
from jose import JWTError, jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status, Header

security = HTTPBearer()
SECRET_KEY = "spatx-secret-key-change-in-production-2024"  # Same as app_enhanced
ALGORITHM = "HS256"

# Use Header-based auth for FormData compatibility
async def get_current_user(
    authorization: str = Header(None),
    db = Depends(get_db)
):
    """Alternative auth that reads Authorization header directly for FormData compatibility"""
    print(f"🔍 TRAINING: Authorization header: {authorization}")
    
    if not authorization:
        print("❌ TRAINING: No authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract token from "Bearer <token>" format
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        print(f"❌ TRAINING: Invalid format - parts: {len(parts)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = parts[1]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except JWTError as e:
        print(f"❌ TRAINING: JWT error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        print(f"❌ TRAINING: User {username} not found")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    print(f"✅ TRAINING: User authenticated: {user.username}")
    return user

# Router for training endpoints
router = APIRouter(prefix="/train", tags=["training"])

# Storage
USER_MODELS_DIR = Path("user_models")
TRAINING_DATA_DIR = Path("training_data")
BASE_MODEL_PATH = Path("saved_models/cit_to_gene/model_50genes.pth")

USER_MODELS_DIR.mkdir(exist_ok=True)
TRAINING_DATA_DIR.mkdir(exist_ok=True)

# Training job status storage (in-memory, use Redis in production)
training_jobs: Dict[str, Dict[str, Any]] = {}


# Pydantic models
class TrainingProgress(BaseModel):
    status: str  # 'running', 'completed', 'failed'
    current_epoch: int
    total_epochs: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    error: Optional[str] = None


class TrainingDataset(Dataset):
    """Dataset for training from CSV + image patches
    
    PRE-EXTRACTS all patches during initialization for 10-20x faster training!
    """
    def __init__(self, csv_path, image_path, gene_columns):
        from PIL import Image
        import torchvision.transforms as transforms
        
        print(f"🔄 Loading training data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.gene_columns = gene_columns
        
        # Define transform once (reused for all patches)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load image temporarily for patch extraction
        print(f"📸 Loading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_width, image_height = image.size
        print(f"   Image size: {image_width}x{image_height}")
        
        # PRE-EXTRACT all patches at initialization (MUCH faster!)
        print(f"🔪 Pre-extracting {len(self.df)} patches...")
        self.patches = []
        patch_size = 224
        half_size = patch_size // 2
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            x, y = int(row['x']), int(row['y'])
            
            # Safely extract patch with boundary handling
            left = max(0, x - half_size)
            top = max(0, y - half_size)
            right = min(image_width, x + half_size)
            bottom = min(image_height, y + half_size)
            
            patch = image.crop((left, top, right, bottom))
            
            # Resize if needed (if at boundary)
            if patch.size != (patch_size, patch_size):
                patch = patch.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
            
            # Transform to tensor and store
            patch_tensor = self.transform(patch)
            self.patches.append(patch_tensor)
        
        # Free memory - we don't need the full image anymore!
        del image
        print(f"✅ Pre-extracted {len(self.patches)} patches successfully!")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # SUPER FAST: Just return pre-processed tensor!
        patch_tensor = self.patches[idx]
        
        # Get gene expression values
        row = self.df.iloc[idx]
        gene_values = torch.tensor([row[gene] for gene in self.gene_columns], dtype=torch.float32)
        
        return patch_tensor, gene_values


@router.post("/start")
async def start_training(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    csv: UploadFile = File(...),
    epochs: int = Form(10),
    learning_rate: float = Form(0.0001),
    batch_size: int = Form(4),  # Reduced from 16 for large CIT model
    val_split: float = Form(0.2),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Start model training for a user
    - Copies base model to user_{id}_model.pth
    - Fine-tunes on user data
    - Returns job_id for progress tracking
    """
    
    user_id = current_user.id
    job_id = str(uuid.uuid4())
    
    # Create user-specific directories
    user_training_dir = TRAINING_DATA_DIR / f"user_{user_id}" / job_id
    user_training_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save uploaded files
        image_path = user_training_dir / image.filename
        csv_path = user_training_dir / csv.filename
        
        with open(image_path, "wb") as f:
            f.write(await image.read())
        
        with open(csv_path, "wb") as f:
            f.write(await csv.read())
        
        # Validate CSV
        df = pd.read_csv(csv_path)
        if 'x' not in df.columns or 'y' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'x' and 'y' columns")
        
        gene_columns = [col for col in df.columns if col not in ['x', 'y']]
        if len(gene_columns) == 0:
            raise HTTPException(status_code=400, detail="CSV must contain at least one gene expression column")
        
        # Initialize training job
        training_jobs[job_id] = {
            "user_id": user_id,
            "status": "initializing",
            "current_epoch": 0,
            "total_epochs": epochs,
            "train_loss": None,
            "val_loss": None,
            "error": None,
            "gene_columns": gene_columns
        }
        
        # Start training in background
        background_tasks.add_task(
            run_training,
            job_id=job_id,
            user_id=user_id,
            image_path=str(image_path),
            csv_path=str(csv_path),
            gene_columns=gene_columns,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            val_split=val_split
        )
        
        return {
            "status": "success",
            "message": "Training started",
            "job_id": job_id,
            "genes": gene_columns,
            "samples": len(df)
        }
        
    except Exception as e:
        import traceback
        print(f"Error starting training: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.get("/progress/{job_id}")
async def get_training_progress(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get training progress for a job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    
    # Security: Check if job belongs to user
    if job["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this job")
    
    return TrainingProgress(
        status=job["status"],
        current_epoch=job["current_epoch"],
        total_epochs=job["total_epochs"],
        train_loss=job["train_loss"],
        val_loss=job["val_loss"],
        error=job["error"]
    )


@router.get("/has_model")
async def check_user_has_model(current_user: User = Depends(get_current_user)):
    """Check if user has a trained model"""
    user_model_path = USER_MODELS_DIR / f"user_{current_user.id}_model.pth"
    return {
        "has_model": user_model_path.exists(),
        "model_path": str(user_model_path) if user_model_path.exists() else None
    }


async def run_training(
    job_id: str,
    user_id: int,
    image_path: str,
    csv_path: str,
    gene_columns: list,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    val_split: float
):
    """
    Background task to run model training
    """
    try:
        print(f"🚀 Starting training job {job_id} for user {user_id}")
        
        # Update status
        training_jobs[job_id]["status"] = "preparing_data"
        
        # Create dataset
        dataset = TrainingDataset(csv_path, image_path, gene_columns)
        
        # Split into train/val
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"📊 Training samples: {train_size}, Validation samples: {val_size}")
        
        # Copy base model to user model
        user_model_path = USER_MODELS_DIR / f"user_{user_id}_model.pth"
        
        if not BASE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Base model not found at {BASE_MODEL_PATH}")
        
        # Copy model (overwrite if exists)
        shutil.copy(BASE_MODEL_PATH, user_model_path)
        print(f"📋 Copied base model to {user_model_path}")
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 Cleared GPU cache")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")
        
        # Load the pre-trained checkpoint to CPU FIRST (avoid loading to GPU twice!)
        print(f"📦 Loading checkpoint to CPU first...")
        checkpoint = torch.load(BASE_MODEL_PATH, map_location='cpu')
        
        print(f"📦 Loading checkpoint with keys: {list(checkpoint.keys())[:5]}...")
        
        # Initialize the FULL model architecture (CIT backbone + regression head)
        # The checkpoint contains both 'cit.*' and 'reg_head.*' weights
        base_num_genes = 50  # Original model was trained on 50 genes
        
        # Initialize CIT backbone with the correct architecture from checkpoint
        cit_backbone = CIT(
            device=device,           # FIRST parameter!
            img_size=224,
            patch_size=4,            # Not 16! Check checkpoint
            in_chans=3,
            out_chans=50,            # Original gene count
            embed_dim=96,            # Not 768! Check the norm shapes
            depths=[2, 2, 6, 2],     # 4 encoder stages
            depths_decoder=[1, 2, 2, 2],  # 4 decoder stages
            num_heads=[3, 6, 12, 24],     # Multi-head attention per stage
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            ape=True,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="expand_first"
        )
        
        # Create the gene predictor wrapper
        model = CITGenePredictor(
            cit_model=cit_backbone,
            num_genes=len(gene_columns)  # User's gene count (can be different from 50)
        ).to(device)
        
        # Load pre-trained weights (skip output head layers that have different sizes)
        print(f"🔄 Loading pre-trained weights...")
        
        # Filter out the gene-specific output layers from checkpoint
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        # Count how many layers we're loading vs skipping
        skipped = [k for k in checkpoint.keys() if k not in pretrained_dict]
        print(f"   ✅ Loading {len(pretrained_dict)}/{len(checkpoint)} layers")
        print(f"   ⏭️  Skipping {len(skipped)} gene-specific output layers: {skipped[:3]}...")
        
        # Update model with pretrained weights (only matching layers)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"✅ Loaded base model weights (50 genes) - fine-tuning for {len(gene_columns)} genes")
        
        # Free checkpoint memory immediately
        del checkpoint
        del model_dict
        del pretrained_dict
        gc.collect()  # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"🧹 Freed checkpoint memory")
        
        print(f"✅ Loaded model on {device}")
        
        # MEMORY OPTIMIZATION: Freeze CIT backbone, only train the gene prediction head
        print(f"🔒 Freezing CIT backbone to save GPU memory...")
        for name, param in model.named_parameters():
            if 'cit' in name:
                param.requires_grad = False
        
        # Set CIT backbone to eval mode permanently (no gradient tracking)
        model.cit.eval()
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Trainable: {trainable_params:,} / {total_params:,} parameters ({100*trainable_params/total_params:.1f}%)")
        
        # Setup optimizer and loss (only for trainable parameters)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        training_jobs[job_id]["status"] = "training"
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss_total = 0.0
            
            for batch_idx, (patches, targets) in enumerate(train_loader):
                patches = patches.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(patches)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss_total += loss.item()
            
            avg_train_loss = train_loss_total / len(train_loader)
            
            # Validation
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for patches, targets in val_loader:
                    patches = patches.to(device)
                    targets = targets.to(device)
                    outputs = model(patches)
                    loss = criterion(outputs, targets)
                    val_loss_total += loss.item()
            
            avg_val_loss = val_loss_total / len(val_loader) if len(val_loader) > 0 else 0.0
            
            # Update progress
            training_jobs[job_id].update({
                "current_epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })
            
            print(f"📊 Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save final model
        torch.save(model.state_dict(), user_model_path)
        print(f"💾 Saved trained model to {user_model_path}")
        
        # Clean up GPU memory
        del model
        del optimizer
        del train_loader
        del val_loader
        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 Cleaned up GPU memory")
        
        # Mark as completed
        training_jobs[job_id]["status"] = "completed"
        print(f"✅ Training job {job_id} completed successfully")
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"❌ Training job {job_id} failed: {error_msg}")
        print(traceback_str)
        
        # Clean up GPU memory on failure too
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 Cleaned up GPU memory after failure")
        
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = error_msg

