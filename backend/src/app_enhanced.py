#!/usr/bin/env python3
"""
Enhanced SpatX Backend - Production Ready
Includes authentication, credits, and model protection
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import shutil
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt

# Import model components
from spatx_core.spatx_core.models.cit_to_gene.CiT_Net_T import CIT
from spatx_core.spatx_core.models.cit_to_gene.CiTGene import CITGenePredictor
from spatx_core.spatx_core.data_adapters import BreastDataAdapter
from spatx_core.spatx_core.trainers import SimpleCITTrainer

# Import database and auth
from database import get_db, User, CreditTransaction
from models import UserCreate, UserLogin, UserResponse, Token, get_operation_cost
from gene_metadata import GENE_INFO, GENE_CATEGORIES, get_gene_info, get_gene_category

# Configuration
UPLOAD_DIR = Path("uploads")
MODELS_DIR = Path("saved_models")
USER_MODELS_DIR = Path("user_models")  # Separate directory for user-trained models
PATCH_SIZE = 224
DEVICE = "cpu"

# Security configuration
SECRET_KEY = "spatx-secret-key-change-in-production-2024"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Load gene set from the new 50-gene model
def load_working_model_genes():
    """Load gene list from model_genes.py (50 genes)"""
    try:
        import sys
        model_dir = os.path.join("spatx_core", "saved_models", "cit_to_gene")
        sys.path.insert(0, model_dir)
        from model_genes import gene_ids
        sys.path.remove(model_dir)
        print(f"[OK] Loaded {len(gene_ids)} genes from model_genes.py")
        return gene_ids
    except ImportError as e:
        print(f"Warning: Could not load model_genes.py ({e}), using fallback genes")
        # Fallback to 50 breast cancer genes
        return ['ABCC11', 'ADH1B', 'ADIPOQ', 'ANKRD30A', 'AQP1', 'AQP3', 'CCR7', 'CD3E', 'CEACAM6', 'CEACAM8', 
                'CLIC6', 'CYTIP', 'DST', 'ERBB2', 'ESR1', 'FASN', 'GATA3', 'IL2RG', 'IL7R', 'KIT', 'KLF5', 
                'KRT14', 'KRT5', 'KRT6B', 'MMP1', 'MMP12', 'MS4A1', 'MUC6', 'MYBPC1', 'MYH11', 'MYLK', 'OPRPN', 
                'OXTR', 'PIGR', 'PTGDS', 'PTN', 'PTPRC', 'SCD', 'SCGB2A1', 'SERHL2', 'SERPINA3', 'SFRP1', 
                'SLAMF7', 'TACSTD2', 'TCL1A', 'TENT5C', 'TOP2A', 'TPSAB1', 'TRAC', 'VWF']

GENE_SET = load_working_model_genes()

app = FastAPI(
    title="SpatX - Spatial Transcriptomics Platform", 
    version="2.0.0",
    description="Professional spatial transcriptomics analysis with authentication and credits"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
USER_MODELS_DIR.mkdir(exist_ok=True)
(MODELS_DIR / "cit_to_gene").mkdir(exist_ok=True)
(USER_MODELS_DIR / "cit_to_gene").mkdir(exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Include training router
from app_training import router as training_router
app.include_router(training_router)

# Include Pratyaksha (Spatial Viewer) router
try:
    from app_pratyaksha import router as pratyaksha_router, PRATYAKSHA_TILES_DIR as TILES_DIR
    app.include_router(pratyaksha_router)
    
    # Mount Pratyaksha tiles as static files (uses path detected by app_pratyaksha)
    if TILES_DIR.exists():
        app.mount("/pratyaksha_tiles", StaticFiles(directory=str(TILES_DIR)), name="pratyaksha_tiles")
        print(f"[OK] Pratyaksha tiles mounted from {TILES_DIR}")
    else:
        # Try fallback paths for local development
        import sys
        if sys.platform == "win32":
            local_tiles = Path(__file__).parent / "Pratyaksha_Base_Code" / "Version_0b" / "tiles"
            if local_tiles.exists():
                app.mount("/pratyaksha_tiles", StaticFiles(directory=str(local_tiles)), name="pratyaksha_tiles")
                print(f"[OK] Pratyaksha tiles mounted from cloned repo: {local_tiles}")
            else:
                print(f"[WARN] Pratyaksha tiles not found. Create pratyaksha_tiles/ folder or clone the repo.")
        else:
            print(f"[WARN] Pratyaksha tiles directory not found: {TILES_DIR}")
    print("[OK] Pratyaksha router loaded")
except ImportError as e:
    print(f"[WARN] Pratyaksha module not available: {e}")

# ============================================================================
# AUTHENTICATION UTILITIES
# ============================================================================

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    token = credentials.credentials
    username = verify_token(token)
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# Alternative auth for endpoints that might have issues with HTTPBearer (like FormData uploads)
from fastapi import Header

async def get_current_user_optional(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """Alternative auth that reads Authorization header directly"""
    print(f"[DEBUG] Authorization header received: {authorization}")
    
    if not authorization:
        print("[ERROR] DEBUG: No authorization header found")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract token from "Bearer <token>" format
    parts = authorization.split()
    print(f"[DEBUG] Authorization parts: {parts}")
    
    if len(parts) != 2 or parts[0].lower() != "bearer":
        print(f"[ERROR] DEBUG: Invalid format - parts count: {len(parts)}, first part: {parts[0] if parts else 'none'}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = parts[1]
    print(f"[DEBUG] Extracted token (first 20 chars): {token[:20]}...")
    
    try:
        username = verify_token(token)
        print(f"[OK] DEBUG: Token verified for user: {username}")
    except Exception as e:
        print(f"[ERROR] DEBUG: Token verification failed: {e}")
        raise
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        print(f"[ERROR] DEBUG: User {username} not found in database")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    print(f"[OK] DEBUG: User authenticated: {user.username} (ID: {user.id})")
    return user

def consume_credits(db: Session, user: User, operation: str, description: str = None):
    """Consume credits for an operation and record transaction"""
    cost = get_operation_cost(operation)
    
    if user.credits < cost:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient credits. Required: {cost}, Available: {user.credits}"
        )
    
    # Deduct credits
    user.credits -= cost
    
    # Record transaction
    transaction = CreditTransaction(
        user_id=user.id,
        operation=operation,
        credits_used=cost,
        credits_remaining=user.credits,
        description=description
    )
    
    db.add(transaction)
    db.commit()
    db.refresh(user)
    
    return user.credits

# ============================================================================
# MODEL UTILITIES (WITH PROTECTION)
# ============================================================================

def create_model(num_genes: int = 50) -> CITGenePredictor:
    """Create CIT model for gene prediction"""
    backbone = CIT(
        device=DEVICE,
        img_size=224, 
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24]
    )
    model = CITGenePredictor(backbone, num_genes=num_genes)
    return model.to(DEVICE)

def load_model(model_path: str) -> Tuple[CITGenePredictor, int]:
    """Load trained model from file and return model + actual gene count"""
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # Check the actual output size from the model
    if 'reg_head.2.weight' in state_dict:
        actual_gene_count = state_dict['reg_head.2.weight'].shape[0]
        print(f"Model was trained for {actual_gene_count} genes")
    else:
        actual_gene_count = 1  # fallback
    
    # Create model with the correct number of genes
    model = create_model(actual_gene_count)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, actual_gene_count

def create_user_model_copy(user_id: int, base_model_path: str) -> str:
    """Create a copy of the base model for user training (protects original)"""
    user_model_dir = USER_MODELS_DIR / "cit_to_gene" / f"user_{user_id}"
    user_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_model_path = user_model_dir / f"model_{timestamp}.pth"
    
    # Copy the base model
    shutil.copy2(base_model_path, user_model_path)
    print(f"Created user model copy: {user_model_path}")
    
    return str(user_model_path)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "SpatX - Spatial Transcriptomics Platform", 
        "version": "2.0.0",
        "status": "running",
        "features": ["authentication", "credits", "model_protection"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "device": DEVICE, 
        "genes": len(GENE_SET),
        "gene_sample": GENE_SET[:5]
    }

@app.get("/genes")
async def get_gene_list():
    """Get the list of available genes"""
    return {"genes": GENE_SET, "count": len(GENE_SET)}

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user already exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        credits=10.0,  # Give 10 free credits to new users
        is_active=True
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token = create_access_token(data={"sub": db_user.username})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": db_user
    }

@app.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    user = db.query(User).filter(User.username == user_data.username).first()
    
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create access token
    access_token = create_access_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user

@app.get("/users/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "credits": current_user.credits,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }

@app.get("/auth/credits")
async def get_credits(current_user: User = Depends(get_current_user)):
    """Get user's current credit balance"""
    return {
        "username": current_user.username,
        "credits": current_user.credits,
        "operation_costs": {
            "training": get_operation_cost("training"),
            "prediction": get_operation_cost("prediction")
        }
    }

# ============================================================================
# FILE UPLOAD ENDPOINTS
# ============================================================================

@app.post("/upload/csv")
async def upload_csv(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_optional)
):
    """Upload CSV file for training or prediction"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Create user-specific directory
    user_dir = UPLOAD_DIR / f"user_{current_user.id}"
    user_dir.mkdir(exist_ok=True)
    
    file_path = user_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Validate CSV structure
    try:
        df = pd.read_csv(file_path)
        return {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "rows": len(df),
            "columns": list(df.columns),
            "user": current_user.username
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

@app.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_optional)
):
    """Upload histology image (WSI)"""
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Create user-specific directory
    user_dir = UPLOAD_DIR / f"user_{current_user.id}"
    user_dir.mkdir(exist_ok=True)
    
    file_path = user_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Get image dimensions and create web-friendly preview for TIFF files
    try:
        img = Image.open(file_path)
        width, height = img.size
        
        web_filename = file.filename
        is_tiff = file.filename.lower().endswith(('.tif', '.tiff'))
        
        if is_tiff:
            # Browsers can't display TIFF - convert to PNG for preview
            png_filename = os.path.splitext(file.filename)[0] + '.png'
            png_path = user_dir / png_filename
            img_rgb = img.convert('RGB')
            img_rgb.save(png_path, 'PNG')
            img_rgb.close()
            web_filename = png_filename
            print(f"[TIFF] Converted {file.filename} to {png_filename} for browser display")
        
        img.close()
        
        return {
            "status": "success",
            "filename": file.filename,
            "web_filename": web_filename,
            "path": str(file_path),
            "dimensions": {"width": width, "height": height},
            "size_mb": round(len(content) / (1024*1024), 2),
            "user": current_user.username
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# ============================================================================
# PREDICTION ENDPOINT (Using working model)
# ============================================================================

@app.post("/predict")
async def predict_genes(
    image_file: str = Form(...),
    coordinates_csv: str = Form(...),
    model_name: str = Form("working_model"),
    output_format: str = Form("json"),
    selected_genes: str = Form("all"),  # Comma-separated gene names or "all"
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predict gene expression from histology image and coordinates"""
    try:
        # Check and consume credits
        remaining_credits = consume_credits(
            db=db, 
            user=current_user, 
            operation="prediction",
            description=f"Prediction on {image_file} with {coordinates_csv}"
        )
        
        # User-specific paths
        user_dir = UPLOAD_DIR / f"user_{current_user.id}"
        image_path = user_dir / image_file
        coords_path = user_dir / coordinates_csv
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found: {image_file}")
        if not coords_path.exists():
            raise HTTPException(status_code=404, detail=f"Coordinates file not found: {coordinates_csv}")
        
        # Load coordinates
        coords_df = pd.read_csv(coords_path)
        required_cols = ['x', 'y']
        if not all(col in coords_df.columns for col in required_cols):
            # Try alternative column names
            x_col = next((col for col in coords_df.columns if 'x' in col.lower()), None)
            y_col = next((col for col in coords_df.columns if 'y' in col.lower()), None)
            if not x_col or not y_col:
                raise HTTPException(status_code=400, detail="CSV must contain x,y coordinate columns")
            coords_df = coords_df.rename(columns={x_col: 'x', y_col: 'y'})
        
        coordinates = [(row['x'], row['y']) for _, row in coords_df.iterrows()]
        
        # Extract patches from WSI (implementation from app_simple.py)
        from app_simple import extract_patches_from_wsi, save_patches, predict_patches
        
        patches = extract_patches_from_wsi(str(image_path), coordinates)
        
        # Save patches temporarily
        patch_dir = user_dir / "temp_patches"
        patch_dir.mkdir(exist_ok=True)
        patch_paths = save_patches(patches, coordinates, f"predict_{current_user.id}", patch_dir)
        
        # Load the 50-gene model ONLY
        working_model_path = Path("saved_models/cit_to_gene/model_50genes.pth")
        if not working_model_path.exists():
            raise HTTPException(status_code=404, detail=f"50-gene model not found at {working_model_path}")
        
        print(f"Loading model from: {working_model_path}")
        model, actual_gene_count = load_model(str(working_model_path))
        print(f"Model loaded: {actual_gene_count} genes")
        
        # Predict gene expression
        predictions = predict_patches(model, patch_paths)
        
        # Clean up temporary patches
        for patch_path in patch_paths:
            os.remove(patch_path)
        
        # Parse selected genes (if user selected specific ones)
        if selected_genes == "all" or not selected_genes:
            genes_to_use = GENE_SET[:actual_gene_count]
        else:
            selected_gene_list = [g.strip() for g in selected_genes.split(',')]
            # Filter to only valid genes that the model can predict
            genes_to_use = [g for g in selected_gene_list if g in GENE_SET[:actual_gene_count]]
            if not genes_to_use:
                genes_to_use = GENE_SET[:actual_gene_count]  # Fallback to all if invalid selection
        
        print(f"Selected genes for heatmap generation: {genes_to_use}")
        
        # Format results - now supporting multi-gene predictions
        results = []
        effective_genes = GENE_SET[:actual_gene_count]  # Model predicts all genes
        
        for i, ((x, y), pred) in enumerate(zip(coordinates, predictions)):
            result = {
                "spot_id": f"spot_{i:04d}",
                "x": float(x),
                "y": float(y)
            }
            # Add predictions for all genes (even if not selected - data is available)
            for j, gene in enumerate(effective_genes):
                result[gene] = float(pred[j])
            results.append(result)
        
        # Calculate statistics for all genes
        gene_stats = {}
        for j, gene in enumerate(effective_genes):
            values = [pred[j] for pred in predictions]
            gene_stats[gene] = {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
        
        # Generate heatmaps for ALL genes
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata
        from matplotlib.colors import LinearSegmentedColormap
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # Get image dimensions (use PNG if TIFF was converted)
        dim_source = image_path
        if str(image_path).lower().endswith(('.tif', '.tiff')):
            png_ver = image_path.parent / (os.path.splitext(image_path.name)[0] + '.png')
            if png_ver.exists():
                dim_source = png_ver
        wsi = Image.open(dim_source)
        img_width, img_height = wsi.size
        wsi.close()
        aspect_ratio = img_width / img_height
        
        # Extract coordinates (same for all genes)
        x_coords = np.array([r['x'] for r in results])
        y_coords = np.array([r['y'] for r in results])
        
        # Custom colormap - Blue -> Cyan -> Yellow -> Red
        colors = ['#08306b', '#2171b5', '#6baed6', '#c6dbef', '#ffffb2', '#fecc5c', '#fd8d3c', '#e31a1c']
        cmap = LinearSegmentedColormap.from_list('clean_spatial', colors, N=256)
        
        # Generate heatmaps ONLY for selected genes (not all 50)
        heatmap_files = []
        print(f"Generating heatmaps for {len(genes_to_use)} selected genes (out of {len(effective_genes)} total)...")
        
        for gene_idx, gene in enumerate(genes_to_use):
            # Get gene values from results (gene must exist in effective_genes)
            if gene not in effective_genes:
                print(f"Warning: Gene {gene} not in model output, skipping")
                continue
            gene_values = np.array([r[gene] for r in results])
            
            # Apply shifted log transformation to handle negative values
            min_val = gene_values.min()
            if min_val < 0:
                # Shift to make all values positive, then apply log1p
                shifted_values = gene_values - min_val + 1
                gene_values_transformed = np.log1p(shifted_values)
                print(f"  {gene}: Applied shifted log transform (original range: [{min_val:.3f}, {gene_values.max():.3f}] → transformed: [{gene_values_transformed.min():.3f}, {gene_values_transformed.max():.3f}])")
            else:
                # All values already positive, just apply log1p
                gene_values_transformed = np.log1p(gene_values)
                print(f"  {gene}: Applied log1p transform (range: [{gene_values.min():.3f}, {gene_values.max():.3f}] → [{gene_values_transformed.min():.3f}, {gene_values_transformed.max():.3f}])")
            
            gene_values = gene_values_transformed  # Use transformed values
            
            # Calculate the actual prediction region (where we have data)
            PATCH_SIZE = 224
            HALF_PATCH = PATCH_SIZE // 2
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Detect the stride (spacing between prediction points)
            x_sorted = sorted(set(x_coords))
            y_sorted = sorted(set(y_coords))
            
            stride_x = x_sorted[1] - x_sorted[0] if len(x_sorted) > 1 else PATCH_SIZE
            stride_y = y_sorted[1] - y_sorted[0] if len(y_sorted) > 1 else PATCH_SIZE
            
            # Extend prediction region by HALF the stride (not half patch)
            # This ensures edge blocks are square, not rectangular
            pred_min_x = max(0, min_x - stride_x // 2)
            pred_max_x = min(img_width, max_x + stride_x // 2)
            pred_min_y = max(0, min_y - stride_y // 2)
            pred_max_y = min(img_height, max_y + stride_y // 2)
            
            pred_width = pred_max_x - pred_min_x
            pred_height = pred_max_y - pred_min_y
            
            print(f"  {gene}: Prediction region [{pred_min_x:.0f}-{pred_max_x:.0f}, {pred_min_y:.0f}-{pred_max_y:.0f}]")
            print(f"  {gene}: Image size [{img_width} x {img_height}], Prediction coverage: {pred_width}x{pred_height}")
            
            # Create high-resolution grid for PREDICTION REGION ONLY
            grid_resolution = 500
            xi = np.linspace(pred_min_x, pred_max_x, grid_resolution)
            yi = np.linspace(pred_min_y, pred_max_y, grid_resolution)
            xi, yi = np.meshgrid(xi, yi)
            
            # Use NEAREST-NEIGHBOR interpolation (no smoothing)
            zi = griddata((x_coords, y_coords), gene_values, (xi, yi), method='nearest')
            
            # Create figure matching PREDICTION REGION size (1:1 with overlay)
            dpi = 100
            fig_width_inches = pred_width / dpi
            fig_height_inches = pred_height / dpi
            fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches), dpi=dpi, facecolor='white')
            
            # Plot heatmap covering PREDICTION REGION
            # Use origin='upper' to match image coordinate system (top-left origin)
            im = ax.imshow(zi, extent=[pred_min_x, pred_max_x, pred_max_y, pred_min_y],
                          origin='upper', cmap=cmap, aspect='equal', interpolation='nearest')
            
            # Set limits to PREDICTION REGION
            ax.set_xlim(pred_min_x, pred_max_x)
            ax.set_ylim(pred_max_y, pred_min_y)  # Inverted Y for image coordinates
            
            # Add colorbar on the right (outside the plot)
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=9, colors='#34495E')
            
            # Minimal styling for clean comparison
            ax.set_title(f'{gene} Expression Heatmap', 
                        fontsize=16, fontweight='bold', pad=10, color='#2C3E50')
            ax.set_xlabel('X Position (pixels)', fontsize=11, fontweight='bold', color='#34495E')
            ax.set_ylabel('Y Position (pixels)', fontsize=11, fontweight='bold', color='#34495E')
            ax.tick_params(labelsize=9, colors='#7F8C8D')
            
            # Remove grid for cleaner look
            ax.grid(False)
            
            # Clean spines
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('#BDC3C7')
                ax.spines[spine].set_linewidth(1)
            
            # Save heatmap as PNG (browsers can't display TIFF)
            image_file_base = os.path.splitext(image_file)[0]
            heatmap_filename = f"heatmap_{gene}_{image_file_base}.png"
            heatmap_path = user_dir / heatmap_filename
            plt.savefig(heatmap_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1, facecolor='white', edgecolor='none')
            plt.close()
            
            heatmap_files.append(heatmap_filename)
            
            # ========================================================================
            # GENERATE OVERLAY IMAGE (tissue + semi-transparent heatmap)
            # ========================================================================
            
            # Load the original tissue image (use PNG version if TIFF was converted)
            tissue_source = image_path
            if str(image_path).lower().endswith(('.tif', '.tiff')):
                png_version = image_path.parent / (os.path.splitext(image_path.name)[0] + '.png')
                if png_version.exists():
                    tissue_source = png_version
            tissue_img = Image.open(tissue_source).convert('RGBA')
            
            # Crop tissue to SAME prediction region as heatmap (perfect 1:1 match)
            tissue_cropped = tissue_img.crop((pred_min_x, pred_min_y, pred_max_x, pred_max_y))
            
            # Create figure matching PREDICTION REGION size (same as heatmap)
            fig_overlay, ax_overlay = plt.subplots(figsize=(pred_width/dpi, pred_height/dpi), 
                                                    dpi=dpi, facecolor='white')
            
            # Display the CROPPED tissue image as background
            # IMPORTANT: Use origin='upper' for PIL images (top-left origin)
            ax_overlay.imshow(tissue_cropped, extent=[pred_min_x, pred_max_x, pred_max_y, pred_min_y], 
                             origin='upper', aspect='equal')
            
            # Overlay the heatmap with transparency - SAME extent and origin as tissue
            im_overlay = ax_overlay.imshow(zi, extent=[pred_min_x, pred_max_x, pred_max_y, pred_min_y],
                                          origin='upper', cmap=cmap, aspect='equal', 
                                          interpolation='nearest', alpha=0.6)  # 60% opacity, blocky like heatmap
            
            # Set limits to PREDICTION REGION (same as heatmap)
            ax_overlay.set_xlim(pred_min_x, pred_max_x)
            ax_overlay.set_ylim(pred_max_y, pred_min_y)  # Inverted Y for image coordinates
            
            # Add colorbar for overlay (same as heatmap view)
            divider_overlay = make_axes_locatable(ax_overlay)
            cax_overlay = divider_overlay.append_axes("right", size="5%", pad=0.1)
            cbar_overlay = plt.colorbar(im_overlay, cax=cax_overlay)
            cbar_overlay.ax.tick_params(labelsize=9, colors='#34495E')
            cbar_overlay.set_label('Log Expression', fontsize=10, color='#34495E')
            
            # Remove axes decorations but keep colorbar
            ax_overlay.set_xticks([])
            ax_overlay.set_yticks([])
            for spine in ax_overlay.spines.values():
                spine.set_visible(False)
            
            # Save overlay image as PNG (browsers can't display TIFF)
            overlay_filename = f"overlay_{gene}_{image_file_base}.png"
            overlay_path = user_dir / overlay_filename
            
            # Save with tight bbox but NO padding to avoid cropping issues
            plt.savefig(overlay_path, dpi=dpi, bbox_inches='tight', pad_inches=0,
                       facecolor='white', edgecolor='none', transparent=False)
            plt.close()
            
            heatmap_files.append(overlay_filename)  # Add overlay to file list
            
            # ========================================================================
            # GENERATE CONTOUR PLOT OVERLAY (tissue + smooth contour lines)
            # ========================================================================
            
            # Create smooth grid for contour interpolation (higher resolution for smoother contours)
            contour_resolution = 300
            xi_smooth = np.linspace(pred_min_x, pred_max_x, contour_resolution)
            yi_smooth = np.linspace(pred_min_y, pred_max_y, contour_resolution)
            xi_smooth, yi_smooth = np.meshgrid(xi_smooth, yi_smooth)
            
            # Use CUBIC interpolation first for smooth contours
            zi_smooth = griddata((x_coords, y_coords), gene_values, (xi_smooth, yi_smooth), method='cubic')
            
            # Fill NaN values with nearest neighbor to cover entire region
            mask = np.isnan(zi_smooth)
            if np.any(mask):
                zi_nearest = griddata((x_coords, y_coords), gene_values, (xi_smooth, yi_smooth), method='nearest')
                zi_smooth[mask] = zi_nearest[mask]
            
            # Create figure matching PREDICTION REGION size (same as heatmap/overlay)
            fig_contour, ax_contour = plt.subplots(figsize=(pred_width/dpi, pred_height/dpi), 
                                                    dpi=dpi, facecolor='white')
            
            # Display the CROPPED tissue image as background
            # Use origin='upper' to match image coordinate system
            ax_contour.imshow(tissue_cropped, extent=[pred_min_x, pred_max_x, pred_max_y, pred_min_y], 
                             origin='upper', aspect='equal', interpolation='bilinear')
            
            # Add filled contours with very subtle transparency (30% opacity for better visibility)
            contour_filled = ax_contour.contourf(xi_smooth, yi_smooth, zi_smooth, 
                                                 levels=20, cmap=cmap, alpha=0.3, extend='both')
            
            # Draw contour lines with WHITE outline + colored fill for maximum visibility
            # First, draw thick white lines as background/outline
            contour_white = ax_contour.contour(xi_smooth, yi_smooth, zi_smooth, 
                                               levels=10, colors='white', linewidths=3.0, 
                                               alpha=1.0)
            
            # Then draw thinner colored lines on top based on expression level
            contour_colored = ax_contour.contour(xi_smooth, yi_smooth, zi_smooth, 
                                                 levels=10, cmap=cmap, linewidths=2.0, 
                                                 alpha=0.95)
            
            # Set limits to PREDICTION REGION (inverted Y for image coordinates)
            ax_contour.set_xlim(pred_min_x, pred_max_x)
            ax_contour.set_ylim(pred_max_y, pred_min_y)
            
            # Add colorbar for contour
            divider_contour = make_axes_locatable(ax_contour)
            cax_contour = divider_contour.append_axes("right", size="5%", pad=0.1)
            cbar_contour = plt.colorbar(contour_filled, cax=cax_contour)
            cbar_contour.ax.tick_params(labelsize=9, colors='#34495E')
            cbar_contour.set_label('Log Expression', fontsize=10, color='#34495E')
            
            # Remove axes decorations but keep colorbar
            ax_contour.set_xticks([])
            ax_contour.set_yticks([])
            for spine in ax_contour.spines.values():
                spine.set_visible(False)
            
            # Save contour overlay as PNG (browsers can't display TIFF)
            contour_filename = f"contour_{gene}_{image_file_base}.png"
            contour_path = user_dir / contour_filename
            
            plt.savefig(contour_path, dpi=dpi, bbox_inches='tight', pad_inches=0,
                       facecolor='white', edgecolor='none', transparent=False)
            plt.close()
            
            heatmap_files.append(contour_filename)  # Add contour to file list
            
            if (gene_idx + 1) % 10 == 0:
                print(f"  Generated {gene_idx + 1}/{len(genes_to_use)} heatmaps + overlays + contours...")
        
        print(f"[OK] Generated {len(heatmap_files)//3} heatmaps + {len(heatmap_files)//3} overlays + {len(heatmap_files)//3} contours for selected genes")
        
        # Prepare gene metadata for selected genes
        gene_metadata = {}
        for gene in genes_to_use:
            gene_metadata[gene] = get_gene_info(gene)
        
        response_data = {
            "status": "success",
            "message": f"Predicted gene expression for {len(coordinates)} spots across {len(effective_genes)} genes",
            "model_used": model_name,
            "num_spots": len(coordinates),
            "num_genes": len(genes_to_use),  # Number of genes user selected
            "genes": genes_to_use,  # Only the selected genes for display
            "all_genes": effective_genes,  # All genes that were predicted
            "predictions": results,  # Contains all gene predictions
            "gene_statistics": gene_stats,
            "gene_metadata": gene_metadata,  # Function, clinical significance, Pearson correlation for each gene
            "gene_categories": {cat: genes for cat, genes in GENE_CATEGORIES.items()},  # Category groupings
            "model_type": "multi_gene" if actual_gene_count > 1 else "single_gene",
            "heatmap_files": heatmap_files,  # Only heatmaps for selected genes
            "credits_used": get_operation_cost("prediction"),
            "remaining_credits": remaining_credits,
            "user": current_user.username
        }
        
        # Save results if requested
        if output_format == "csv":
            results_df = pd.DataFrame(results)
            results_path = user_dir / f"predictions_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_df.to_csv(results_path, index=False)
            response_data["results_file"] = str(results_path)
        
        return response_data
        
    except Exception as e:
        import traceback
        error_detail = f"Prediction failed: {str(e)}"
        print(f"ERROR in /predict endpoint: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_detail)

if __name__ == "__main__":
    print("🚀 Starting SpatX Enhanced Platform...")
    print(f"Device: {DEVICE}")
    print(f"Available genes: {len(GENE_SET)}")
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"User models directory: {USER_MODELS_DIR}")
    print("🔐 Authentication: ENABLED")
    print("💰 Credits system: ENABLED") 
    print("[SECURITY] Model protection: ENABLED")
    print("🎓 Training portal: ENABLED")
    print("Server will start on http://localhost:9001")
    print("API docs available at http://localhost:9001/docs")
    print("Press Ctrl+C to stop")
    
    uvicorn.run("app_enhanced:app", host="0.0.0.0", port=9001, reload=True)

