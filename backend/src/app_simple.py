#!/usr/bin/env python3
"""
Simplified SpatX Backend - Clean Implementation
Focuses on core functionality: Training and Prediction for spatial transcriptomics
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import model components
from spatx_core.spatx_core.models.cit_to_gene.CiT_Net_T import CIT
from spatx_core.spatx_core.models.cit_to_gene.CiTGene import CITGenePredictor
from spatx_core.spatx_core.data_adapters import BreastDataAdapter
from spatx_core.spatx_core.trainers import SimpleCITTrainer

# Configuration
UPLOAD_DIR = Path("uploads")
MODELS_DIR = Path("saved_models")
PATCH_SIZE = 224
DEVICE = "cpu"  # Change to "cuda" if GPU available

# Load gene set from working model
def load_working_model_genes():
    """Load gene list from working_model.py"""
    try:
        import sys
        import os
        model_dir = os.path.join("spatx_core", "saved_models", "cit_to_gene")
        sys.path.insert(0, model_dir)
        from working_model import gene_ids
        sys.path.remove(model_dir)
        return gene_ids
    except ImportError:
        print("Warning: Could not load working_model.py, using fallback genes")
        return ["ERBB2", "ESR1", "KIT", "GATA3", "KRT14"]  # Fallback minimal set

GENE_SET = load_working_model_genes()

app = FastAPI(title="SpatX Core API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
(MODELS_DIR / "cit_to_gene").mkdir(exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_patches_from_wsi(wsi_path: str, coordinates: List[Tuple[float, float]], 
                           patch_size: int = PATCH_SIZE) -> List[np.ndarray]:
    """Extract patches from whole slide image at given coordinates.
    
    For large images (>100MP), uses tifffile for memory-efficient region reading
    instead of loading the entire image into RAM.
    """
    # Try to open and get dimensions
    W, H = None, None
    use_tifffile = False
    
    # First try PIL to get size (works for most formats)
    try:
        img_probe = Image.open(wsi_path)
        W, H = img_probe.size
        img_probe.close()
    except Exception as e:
        print(f"PIL couldn't read image header: {e}")
    
    # For TIFF files, try tifffile as it handles more compression formats
    is_tiff = wsi_path.lower().endswith(('.tif', '.tiff'))
    
    if is_tiff:
        try:
            import tifffile
            tif = tifffile.TiffFile(wsi_path)
            page = tif.pages[0]
            if W is None:
                H, W = page.shape[:2]
            total_pixels = W * H
            
            if total_pixels > 100_000_000:
                use_tifffile = True
                print(f"  Using tifffile for memory-efficient patch extraction (large image)")
            else:
                # Small TIFF - load via tifffile to handle all compression types
                tif_array = page.asarray()
                if tif_array.ndim == 2:
                    tif_array = np.stack([tif_array]*3, axis=-1)
                elif tif_array.shape[2] == 4:
                    tif_array = tif_array[:, :, :3]
                wsi = Image.fromarray(tif_array)
                H, W = tif_array.shape[:2]
                use_tifffile = 'tifffile_full'
                tif.close()
                print(f"  Loaded TIFF via tifffile ({W}x{H})")
        except ImportError:
            print(f"  tifffile not installed, falling back to PIL")
        except Exception as e:
            print(f"  tifffile failed: {e}, falling back to PIL")
            use_tifffile = False
    
    if W is None:
        raise ValueError(f"Failed to load image: {wsi_path}")
    
    total_pixels = W * H
    print(f"WSI dimensions: {W}x{H} ({total_pixels/1e6:.0f} MP), extracting {len(coordinates)} patches of size {patch_size}x{patch_size}")
    
    # For large TIFFs with tifffile available, use lazy reading
    if use_tifffile is True:
        print(f"  Using tifffile lazy reading for large image")
    elif use_tifffile == 'tifffile_full':
        pass  # Already loaded above
    elif is_tiff and use_tifffile is False:
        # Try pyvips for TIFFs that PIL can't handle
        try:
            import pyvips
            vips_img = pyvips.Image.new_from_file(wsi_path, access="random")
            use_tifffile = 'pyvips'
            print(f"  Using pyvips for TIFF reading")
        except (ImportError, Exception):
            pass
    
    # Final fallback: load full image with PIL
    if not use_tifffile:
        try:
            wsi = Image.open(wsi_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image: {wsi_path} - {str(e)}")
    
    patches = []
    
    for idx, (x, y) in enumerate(coordinates):
        half_patch = patch_size // 2
        
        x0 = max(0, int(x) - half_patch)
        y0 = max(0, int(y) - half_patch)
        x1 = min(W, int(x) + half_patch)
        y1 = min(H, int(y) + half_patch)
        
        patch_w = x1 - x0
        patch_h = y1 - y0
        
        if patch_w <= 0 or patch_h <= 0:
            print(f"Warning: Invalid patch dimensions at ({x}, {y}): {patch_w}x{patch_h}, skipping")
            continue
        
        try:
            if use_tifffile == 'pyvips':
                region = vips_img.crop(x0, y0, patch_w, patch_h)
                patch_np = np.ndarray(
                    buffer=region.write_to_memory(),
                    dtype=np.uint8,
                    shape=[region.height, region.width, region.bands]
                )
                if patch_np.shape[2] == 4:
                    patch_np = patch_np[:, :, :3]
                patch = Image.fromarray(patch_np)
            elif use_tifffile is True:
                # tifffile lazy: read only the needed region
                region = tif.pages[0].asarray()[y0:y1, x0:x1]
                if region.ndim == 2:
                    region = np.stack([region]*3, axis=-1)
                elif region.shape[2] == 4:
                    region = region[:, :, :3]
                patch = Image.fromarray(region)
            else:
                # PIL or tifffile_full: crop from already-loaded image
                patch = wsi.crop((x0, y0, x1, y1))
            
            if patch.mode != 'RGB':
                patch = patch.convert('RGB')
            
            if patch.size != (patch_size, patch_size):
                patch = patch.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
            
            patches.append(np.array(patch))
            
            if idx < 3:
                print(f"  Patch {idx}: coord=({x},{y}), box=({x0},{y0},{x1},{y1}), size={patch.size}")
                
        except Exception as e:
            print(f"Error extracting patch at ({x}, {y}): {e}")
            raise
    
    # Cleanup
    if use_tifffile == 'pyvips':
        del vips_img
    elif use_tifffile:
        tif.close()
    
    print(f"Successfully extracted {len(patches)} patches")
    return patches

def save_patches(patches: List[np.ndarray], coordinates: List[Tuple[float, float]], 
                wsi_id: str, output_dir: Path) -> List[str]:
    """Save patches as individual PNG files"""
    patch_paths = []
    for i, (patch, (x, y)) in enumerate(zip(patches, coordinates)):
        filename = f"patch_{wsi_id}_{i:04d}_{int(x)}_{int(y)}.png"
        filepath = output_dir / filename
        Image.fromarray(patch).save(filepath)
        patch_paths.append(str(filepath))
    return patch_paths

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
    # First, load the state dict to check the actual number of genes
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

def predict_patches(model: CITGenePredictor, patch_paths: List[str]) -> np.ndarray:
    """Predict gene expression for a list of patch images"""
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    predictions = []
    with torch.no_grad():
        for patch_path in patch_paths:
            # Load and transform patch
            patch = Image.open(patch_path).convert('RGB')
            tensor = transform(patch).unsqueeze(0).to(DEVICE)
            
            # Predict
            pred = model(tensor)
            predictions.append(pred.cpu().numpy()[0])
    
    return np.array(predictions)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"message": "SpatX Core API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": DEVICE, "genes": len(GENE_SET)}

@app.get("/genes")
async def get_gene_list():
    """Get the list of available genes"""
    return {"genes": GENE_SET, "count": len(GENE_SET)}

# ============================================================================
# TRAINING ENDPOINTS
# ============================================================================

@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file for training or prediction"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    file_path = UPLOAD_DIR / file.filename
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
            "columns": list(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload histology image (WSI)"""
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Get image dimensions
    try:
        with Image.open(file_path) as img:
            width, height = img.size
        
        return {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "dimensions": {"width": width, "height": height},
            "size_mb": round(len(content) / (1024*1024), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

@app.post("/train")
async def train_model(
    csv_file: str = Form(...),
    image_file: str = Form(...),
    wsi_ids: str = Form(...),  # comma-separated
    num_epochs: int = Form(10),
    batch_size: int = Form(8),
    learning_rate: float = Form(0.001)
):
    """Train CIT model on uploaded data"""
    try:
        # Validate files exist
        csv_path = UPLOAD_DIR / csv_file
        image_path = UPLOAD_DIR / image_file
        
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_file}")
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found: {image_file}")
        
        # Parse WSI IDs
        wsi_id_list = [wsi.strip() for wsi in wsi_ids.split(',')]
        
        # Create data adapter
        adapter = BreastDataAdapter(
            image_dir=str(UPLOAD_DIR),
            breast_csv=str(csv_path),
            wsi_ids=wsi_id_list,
            gene_ids=GENE_SET
        )
        
        # Create trainer
        trainer = SimpleCITTrainer(
            train_adapter=adapter,
            validation_adapter=adapter,  # Use same data for validation (demo)
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=DEVICE
        )
        
        # Train model
        model, results = trainer.train()
        
        # Save model
        model_name = f"spatx_model_{len(wsi_id_list)}wsi_{num_epochs}epochs"
        model_path = MODELS_DIR / "cit_to_gene" / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "wsi_ids": wsi_id_list,
            "gene_ids": GENE_SET,
            "num_genes": len(GENE_SET),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "best_val_loss": results.best_val_loss,
            "best_epoch": results.best_epoch,
            "training_data": csv_file,
            "image_data": image_file
        }
        
        metadata_path = MODELS_DIR / "cit_to_gene" / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "status": "success",
            "message": "Training completed successfully",
            "model_name": model_name,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "results": {
                "best_val_loss": results.best_val_loss,
                "best_epoch": results.best_epoch,
                "num_genes": len(GENE_SET),
                "num_wsi": len(wsi_id_list)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.post("/predict")
async def predict_genes(
    image_file: str = Form(...),
    coordinates_csv: str = Form(...),
    model_name: str = Form("spatx_model_latest"),
    output_format: str = Form("json")  # json or csv
):
    """Predict gene expression from histology image and coordinates"""
    try:
        # Validate files
        image_path = UPLOAD_DIR / image_file
        coords_path = UPLOAD_DIR / coordinates_csv
        
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
        
        # Extract patches from WSI
        patches = extract_patches_from_wsi(str(image_path), coordinates)
        
        # Save patches temporarily
        patch_dir = UPLOAD_DIR / "temp_patches"
        patch_dir.mkdir(exist_ok=True)
        patch_paths = save_patches(patches, coordinates, "predict", patch_dir)
        
        # Load model - check for working model first
        model_path = Path("spatx_core/saved_models/cit_to_gene/model_working_model.pth")
        if not model_path.exists():
            # Try user-specified model
            model_path = MODELS_DIR / "cit_to_gene" / f"{model_name}.pth"
            if not model_path.exists():
                # Try to find any available model
                model_files = list((MODELS_DIR / "cit_to_gene").glob("*.pth"))
                spatx_model_files = list(Path("spatx_core/saved_models/cit_to_gene").glob("*.pth"))
                all_models = model_files + spatx_model_files
                if not all_models:
                    raise HTTPException(status_code=404, detail="No trained models found. Please train a model first.")
                model_path = all_models[0]  # Use the first available model
        
        model, actual_gene_count = load_model(str(model_path))
        
        # Predict gene expression
        predictions = predict_patches(model, patch_paths)
        
        # Handle single vs multi-gene models
        if actual_gene_count == 1:
            # Single gene model - we need to determine which gene it predicts
            # For now, assume it's the first gene in our list
            predicted_gene = GENE_SET[0]  # Use first gene as default
            print(f"Single-gene model detected. Assuming it predicts: {predicted_gene}")
        else:
            predicted_gene = None
        
        # Clean up temporary patches
        for patch_path in patch_paths:
            os.remove(patch_path)
        
        # Format results
        results = []
        if actual_gene_count == 1:
            # Single gene model
            for i, ((x, y), pred) in enumerate(zip(coordinates, predictions)):
                result = {
                    "spot_id": f"spot_{i:04d}",
                    "x": float(x),
                    "y": float(y),
                    predicted_gene: float(pred[0])  # Single prediction value
                }
                results.append(result)
            
            # Calculate statistics for single gene
            values = [pred[0] for pred in predictions]
            gene_stats = {
                predicted_gene: {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
            }
            effective_genes = [predicted_gene]
        else:
            # Multi-gene model
            for i, ((x, y), pred) in enumerate(zip(coordinates, predictions)):
                result = {
                    "spot_id": f"spot_{i:04d}",
                    "x": float(x),
                    "y": float(y)
                }
                # Add gene predictions
                for j, gene in enumerate(GENE_SET[:actual_gene_count]):
                    result[gene] = float(pred[j])
                results.append(result)
            
            # Calculate statistics
            gene_stats = {}
            for j, gene in enumerate(GENE_SET[:actual_gene_count]):
                values = [pred[j] for pred in predictions]
                gene_stats[gene] = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
            effective_genes = GENE_SET[:actual_gene_count]
        
        response_data = {
            "status": "success",
            "message": f"Predicted gene expression for {len(coordinates)} spots",
            "model_used": model_name,
            "num_spots": len(coordinates),
            "num_genes": len(effective_genes),
            "genes": effective_genes,
            "predictions": results,
            "gene_statistics": gene_stats,
            "model_type": "single_gene" if actual_gene_count == 1 else "multi_gene",
            "actual_gene_count": actual_gene_count
        }
        
        # Save results if requested
        if output_format == "csv":
            results_df = pd.DataFrame(results)
            results_path = UPLOAD_DIR / f"predictions_{model_name}.csv"
            results_df.to_csv(results_path, index=False)
            response_data["results_file"] = str(results_path)
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available trained models"""
    models = []
    
    try:
        # Check for working model first
        working_model_path = Path("spatx_core/saved_models/cit_to_gene/model_working_model.pth")
        if working_model_path.exists():
            try:
                size_mb = round(working_model_path.stat().st_size / (1024*1024), 2)
                models.append({
                    "name": "working_model",
                    "path": str(working_model_path),
                    "size_mb": size_mb,
                    "metadata": {
                        "description": "Pre-trained working model",
                        "genes": len(GENE_SET),
                        "gene_list": GENE_SET[:10] + ["..."] if len(GENE_SET) > 10 else GENE_SET
                    }
                })
            except Exception as e:
                print(f"Error processing working model: {e}")
        
        # Check for user-trained models
        models_dir = MODELS_DIR / "cit_to_gene"
        if models_dir.exists():
            try:
                model_files = list(models_dir.glob("*.pth"))
                
                for model_file in model_files:
                    try:
                        metadata_file = model_file.with_suffix('').with_suffix('_metadata.json')
                        metadata = {}
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                        
                        models.append({
                            "name": model_file.stem,
                            "path": str(model_file),
                            "size_mb": round(model_file.stat().st_size / (1024*1024), 2),
                            "metadata": metadata
                        })
                    except Exception as e:
                        print(f"Error processing model {model_file}: {e}")
                        continue
            except Exception as e:
                print(f"Error scanning models directory: {e}")
        
        return {"models": models, "count": len(models)}
        
    except Exception as e:
        print(f"Error in list_models: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

# ============================================================================
# HEATMAP GENERATION
# ============================================================================

@app.post("/generate_heatmap")
async def generate_heatmap(
    predictions_file: str = Form(...),
    gene_name: str = Form(...),
    colormap: str = Form("RdBu_r")  # Blue-white-red like your image
):
    """Generate heatmap visualization for a specific gene"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from scipy.interpolate import griddata
        
        # Load predictions
        pred_path = UPLOAD_DIR / predictions_file
        if not pred_path.exists():
            raise HTTPException(status_code=404, detail=f"Predictions file not found: {predictions_file}")
        
        df = pd.read_csv(pred_path)
        
        if gene_name not in df.columns:
            available_genes = [col for col in df.columns if col in GENE_SET]
            raise HTTPException(
                status_code=400, 
                detail=f"Gene '{gene_name}' not found. Available genes: {available_genes}"
            )
        
        # Extract coordinates and values
        x = df['x'].values
        y = df['y'].values
        values = df[gene_name].values
        
        # Create heatmap - handle small datasets
        plt.figure(figsize=(12, 10), facecolor='white')
        
        if len(x) < 4:
            # For small datasets, create a simple scatter plot with color
            scatter = plt.scatter(x, y, c=values, cmap=colormap, s=200, alpha=0.9, edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, label=f'Log Expression', shrink=0.8)
        else:
            # For larger datasets, use interpolation
            xi = np.linspace(x.min(), x.max(), 100)  # Higher resolution
            yi = np.linspace(y.min(), y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)
            
            # Interpolate values with error handling
            try:
                zi = griddata((x, y), values, (xi, yi), method='linear')
                # Create smooth heatmap like your image
                plt.imshow(zi, extent=[x.min(), x.max(), y.min(), y.max()], 
                          cmap=colormap, aspect='auto', origin='lower', interpolation='bilinear')
                cbar = plt.colorbar(label=f'Log Expression', shrink=0.8)
                # Add original points as small black dots
                plt.scatter(x, y, c='black', s=10, alpha=0.6)
            except Exception as interp_error:
                print(f"Interpolation failed, using scatter plot: {interp_error}")
                scatter = plt.scatter(x, y, c=values, cmap=colormap, s=200, alpha=0.9, edgecolors='black', linewidth=0.5)
                cbar = plt.colorbar(scatter, label=f'Log Expression', shrink=0.8)
        
        # Style like your image
        plt.title(f'Predicted Expression\n{gene_name}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('X Position', fontsize=14)
        plt.ylabel('Y Position', fontsize=14)
        plt.grid(False)  # Remove grid for cleaner look
        
        # Set colorbar font size
        cbar.ax.tick_params(labelsize=12)
        
        # Save heatmap
        heatmap_path = UPLOAD_DIR / f"heatmap_{gene_name}_{predictions_file.replace('.csv', '.png')}"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "status": "success",
            "gene": gene_name,
            "heatmap_path": str(heatmap_path),
            "heatmap_url": f"/uploads/{heatmap_path.name}",
            "stats": {
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "mean": float(np.nanmean(values)),
                "num_spots": len(values)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")

if __name__ == "__main__":
    print("Starting SpatX Core API...")
    print(f"Device: {DEVICE}")
    print(f"Available genes: {len(GENE_SET)}")
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print("Server will start on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    
    uvicorn.run("app_simple:app", host="0.0.0.0", port=8000, reload=True)
