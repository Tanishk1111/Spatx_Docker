#!/usr/bin/env python3
"""
Pratyaksha - Spatial Transcriptomics Interactive Viewer Backend
Integrated with SpatX Platform
"""
import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
from typing import Optional, List
from pathlib import Path
from PIL import Image as PILImage
# Allow large images (Visium full-res can be 20000+ pixels)
PILImage.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
import shutil

# Scanpy and analysis libraries (optional for local testing)
try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    print("[WARN] scanpy not installed - Expression/DGE analysis will be disabled")

# GO enrichment (optional - graceful fallback if not installed)
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    print("[WARN] gseapy not installed - GO enrichment will be disabled")

# Matplotlib for GO enrichment plots
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt

# ============================================
# Configuration - Auto-detect Local vs Server
# ============================================
def detect_environment():
    """Detect if running locally (Windows), Docker, or on server (Linux)"""
    # Environment variables override everything (used by Docker)
    env_data = os.environ.get("PRATYAKSHA_DATA_DIR")
    env_tiles = os.environ.get("PRATYAKSHA_TILES_DIR")
    if env_data and env_tiles:
        return Path(env_data), Path(env_tiles), "docker"

    if sys.platform == "win32":
        base_dir = Path(__file__).parent.absolute()
        data_dir = base_dir / "pratyaksha_data"
        tiles_dir = base_dir / "pratyaksha_tiles"
        
        cloned_tiles = base_dir / "Pratyaksha_Base_Code" / "Version_0b" / "tiles"
        if cloned_tiles.exists() and not tiles_dir.exists():
            tiles_dir = cloned_tiles
            print(f"[TILES] Using tiles from cloned repo: {tiles_dir}")
        
        return data_dir, tiles_dir, "local"
    else:
        # Check if running in Docker (relative /app paths) or lab server
        app_data = Path("/app/pratyaksha_data")
        if app_data.exists():
            return app_data, Path("/app/pratyaksha_tiles"), "docker"
        return (
            Path("/DATA4/Tanishk/spatx_deployment/pratyaksha_data"),
            Path("/DATA4/Tanishk/spatx_deployment/pratyaksha_tiles"),
            "server"
        )

PRATYAKSHA_DATA_DIR, PRATYAKSHA_TILES_DIR, ENV_MODE = detect_environment()
print(f"[PRATYAKSHA] Running in {ENV_MODE} mode")
print(f"   Data dir: {PRATYAKSHA_DATA_DIR}")
print(f"   Tiles dir: {PRATYAKSHA_TILES_DIR}")

# Default dataset (can be extended to support multiple datasets)
DEFAULT_H5_FILE = PRATYAKSHA_DATA_DIR / "Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"

# User session storage
USER_SESSIONS_DIR = Path(__file__).parent / "pratyaksha_sessions"
USER_SESSIONS_DIR.mkdir(exist_ok=True)

# pyvips for tile generation (optional)
try:
    import pyvips
    PYVIPS_AVAILABLE = True
except ImportError:
    PYVIPS_AVAILABLE = False
    print("[WARN] pyvips not installed - Tile generation will use fallback method")

# ============================================
# Router Setup
# ============================================
router = APIRouter(prefix="/pratyaksha", tags=["Pratyaksha Viewer"])

# ============================================
# Global AnnData Cache
# ============================================
_adata_cache = {}
_demo_mode = False  # Set to True if no data available

# ============================================
# Cell Type Annotation Storage & Status
# ============================================
_annotation_status = {}  # session_id -> {"status": "pending/running/complete/error", "message": "..."}

# Default marker genes for common cell types (breast cancer / general tissue)
# Users can customize this via the API
DEFAULT_CELL_TYPE_MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "MUC1"],
    "Luminal": ["ESR1", "PGR", "GATA3", "FOXA1", "XBP1"],
    "Basal": ["KRT5", "KRT14", "KRT17", "TP63", "EGFR"],
    "HER2+": ["ERBB2", "GRB7", "PGAP3", "STARD3"],
    "T cells": ["CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B"],
    "B cells": ["CD19", "CD79A", "CD79B", "MS4A1", "PAX5"],
    "Macrophages": ["CD68", "CD163", "CSF1R", "MARCO", "MSR1"],
    "Fibroblasts": ["COL1A1", "COL1A2", "COL3A1", "DCN", "FAP", "PDGFRA"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "ERG", "FLT1"],
    "Adipocytes": ["ADIPOQ", "LEP", "FABP4", "PPARG", "LPL"],
    "Smooth Muscle": ["ACTA2", "MYH11", "TAGLN", "CNN1", "DES"],
    "NK cells": ["NCAM1", "NKG7", "GNLY", "PRF1", "KLRD1"],
    "Dendritic cells": ["CD1C", "CLEC9A", "ITGAX", "CD83", "CD86"],
    "Plasma cells": ["SDC1", "IGHG1", "IGHA1", "MZB1", "XBP1"],
}

# Cell type colors for visualization
CELL_TYPE_COLORS = {
    "Epithelial": "#e41a1c",
    "Luminal": "#377eb8",
    "Basal": "#4daf4a",
    "HER2+": "#984ea3",
    "T cells": "#ff7f00",
    "B cells": "#ffff33",
    "Macrophages": "#a65628",
    "Fibroblasts": "#f781bf",
    "Endothelial": "#999999",
    "Adipocytes": "#66c2a5",
    "Smooth Muscle": "#fc8d62",
    "NK cells": "#8da0cb",
    "Dendritic cells": "#e78ac3",
    "Plasma cells": "#a6d854",
    "Unknown": "#cccccc",
}

# ============================================
# Gene Set Variation Analysis (GSVA) Gene Sets
# MSigDB Hallmark gene sets (curated subsets)
# ============================================
GSVA_GENE_SETS = {
    "Hypoxia": [
        "VEGFA", "SLC2A1", "LDHA", "PGK1", "PDK1", "BNIP3", "P4HA1", "ADM",
        "NDRG1", "CA9", "ENO1", "ALDOA", "GAPDH", "HK2", "PFKFB3", "PKM",
        "LOX", "EGLN3", "SLC16A3", "ANGPTL4", "ERO1A", "GPI", "TPI1",
        "BHLHE40", "BNIP3L", "DDIT4", "HIF1A", "HMOX1", "MXI1", "PPP1R15A"
    ],
    "EMT": [
        "VIM", "CDH2", "FN1", "SNAI1", "SNAI2", "TWIST1", "TWIST2", "ZEB1",
        "ZEB2", "MMP2", "MMP9", "SERPINE1", "TGFBI", "COL1A1", "COL1A2",
        "COL3A1", "COL5A1", "COL5A2", "SPARC", "POSTN", "CTGF", "ACTA2",
        "TAGLN", "FAP", "THY1", "CALD1", "FBN1", "DCN", "LUM", "BGN"
    ],
    "Angiogenesis": [
        "VEGFA", "VEGFB", "VEGFC", "FLT1", "KDR", "FLT4", "ANGPT1", "ANGPT2",
        "TEK", "PECAM1", "CDH5", "CXCL8", "HIF1A", "NRP1", "NRP2", "PDGFB",
        "PDGFRB", "THBS1", "THBS2", "SERPINE1", "MMP2", "MMP14", "DLL4",
        "NOTCH1", "ENG", "EFNB2", "EPHB4", "VWF", "MCAM", "ROBO4"
    ],
}

GSVA_COLORS = {
    "Hypoxia": {"low": "#2563eb", "high": "#dc2626"},
    "EMT": {"low": "#059669", "high": "#d97706"},
    "Angiogenesis": {"low": "#6d28d9", "high": "#e11d48"},
}

def get_adata(h5_path: Path = None, session_id: str = None):
    """Load and cache AnnData object. Supports session-based data."""
    global _adata_cache, _demo_mode
    
    if not SCANPY_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Scanpy not installed. Expression analysis unavailable. Install with: pip install scanpy"
        )
    
    # Determine H5 path based on session_id or default
    if session_id:
        session_h5 = USER_SESSIONS_DIR / session_id / "expression.h5"
        if session_h5.exists():
            h5_path = session_h5
            print(f"[SESSION] Using session H5: {h5_path}")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No expression data found for session {session_id}. Did you upload an H5 file?"
            )
    elif h5_path is None:
        h5_path = DEFAULT_H5_FILE
    
    key = str(h5_path)
    if key not in _adata_cache:
        if not h5_path.exists():
            _demo_mode = True
            raise HTTPException(
                status_code=404, 
                detail=f"Data file not found: {h5_path}. The viewer will work but expression analysis is unavailable. Upload Visium .h5 file to {PRATYAKSHA_DATA_DIR}"
            )
        try:
            adata = sc.read_10x_h5(str(h5_path))
            adata.var_names_make_unique()
            _adata_cache[key] = adata
            print(f"[OK] Pratyaksha: Loaded {adata.n_obs} barcodes, {adata.n_vars} genes from {h5_path.name}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load AnnData: {e}")
    
    return _adata_cache[key]

# ============================================
# Request/Response Models
# ============================================
class ExpressionRequest(BaseModel):
    barcodes: List[str]
    genes: Optional[List[str]] = None
    session_id: Optional[str] = None  # For session-based data

class ExpressionAllRequest(BaseModel):
    genes: Optional[List[str]] = None
    session_id: Optional[str] = None  # For session-based data

class DGERequest(BaseModel):
    group1: List[str]
    group2: Optional[List[str]] = None
    comparison_type: Optional[str] = "auto"
    vs_rest: Optional[bool] = False
    session_id: Optional[str] = None  # For session-based data

class GOEnrichmentRequest(BaseModel):
    gene_list: List[str]
    organism: str = "Human"
    ontology_types: List[str] = [
        "GO_Biological_Process_2021",
        "GO_Molecular_Function_2021",
        "GO_Cellular_Component_2021"
    ]
    pvalue_threshold: float = 0.05
    top_terms: Optional[int] = None
    session_id: Optional[str] = None  # For session-based data

class GSVARequest(BaseModel):
    gene_sets: Optional[List[str]] = None  # e.g. ["Hypoxia", "EMT", "Angiogenesis"], None = all
    barcodes: Optional[List[str]] = None   # subset of barcodes, None = all in-tissue
    session_id: Optional[str] = None

# ============================================
# Utility Functions
# ============================================
def sanitize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN, inf, -inf with safe values for JSON serialization."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

# ============================================
# Cell Type Annotation Functions
# ============================================
def run_cell_type_annotation(session_id: str, markers: dict = None):
    """
    Run marker-based cell type annotation for a session.
    This function runs as a background task.
    """
    global _annotation_status
    
    if markers is None:
        markers = DEFAULT_CELL_TYPE_MARKERS
    
    try:
        _annotation_status[session_id] = {"status": "running", "message": "Loading expression data..."}
        
        session_dir = USER_SESSIONS_DIR / session_id
        h5_path = session_dir / "expression.h5"
        
        if not h5_path.exists():
            _annotation_status[session_id] = {
                "status": "error", 
                "message": "No expression data found. Cell type annotation requires expression H5 file."
            }
            return
        
        # Load expression data
        _annotation_status[session_id] = {"status": "running", "message": "Loading expression matrix..."}
        adata = sc.read_10x_h5(str(h5_path))
        
        # Load barcode positions
        barcodes_path = session_dir / "tiles" / "barcodes_fullres.json"
        if not barcodes_path.exists():
            barcodes_path = session_dir / "barcodes.json"
        
        if barcodes_path.exists():
            with open(barcodes_path) as f:
                barcode_coords = json.load(f)
            barcode_to_coord = {b["barcode"]: (b["x"], b["y"]) for b in barcode_coords}
        else:
            barcode_to_coord = {}
        
        _annotation_status[session_id] = {"status": "running", "message": "Normalizing data..."}
        
        # Normalize the data for scoring
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)
        
        _annotation_status[session_id] = {"status": "running", "message": "Scoring cell types..."}
        
        # Score each cell type based on marker genes
        cell_type_scores = {}
        available_genes = set(adata_norm.var_names)
        
        for cell_type, marker_genes in markers.items():
            # Find which markers are in the dataset
            present_markers = [g for g in marker_genes if g in available_genes]
            
            if len(present_markers) >= 1:
                # Calculate mean expression of marker genes per cell
                marker_idx = [adata_norm.var_names.get_loc(g) for g in present_markers]
                
                if hasattr(adata_norm.X, 'toarray'):
                    marker_expr = adata_norm.X[:, marker_idx].toarray()
                else:
                    marker_expr = adata_norm.X[:, marker_idx]
                
                # Mean expression across markers
                cell_type_scores[cell_type] = np.mean(marker_expr, axis=1)
            else:
                cell_type_scores[cell_type] = np.zeros(adata_norm.n_obs)
        
        _annotation_status[session_id] = {"status": "running", "message": "Assigning cell types..."}
        
        # Assign cell type based on highest score
        score_matrix = np.column_stack([cell_type_scores[ct] for ct in cell_type_scores.keys()])
        cell_types_list = list(cell_type_scores.keys())
        
        # Get the cell type with max score for each cell
        max_scores = np.max(score_matrix, axis=1)
        max_indices = np.argmax(score_matrix, axis=1)
        
        # Assign "Unknown" if max score is below threshold
        threshold = 0.1  # Minimum expression threshold
        annotations = []
        
        for i, barcode in enumerate(adata_norm.obs_names):
            if max_scores[i] < threshold:
                cell_type = "Unknown"
                confidence = 0.0
            else:
                cell_type = cell_types_list[max_indices[i]]
                # Confidence = how much higher the top score is vs second highest
                sorted_scores = np.sort(score_matrix[i])[::-1]
                if sorted_scores[1] > 0:
                    confidence = min(1.0, (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0])
                else:
                    confidence = 1.0 if sorted_scores[0] > 0 else 0.0
            
            annotation = {
                "barcode": barcode,
                "cell_type": cell_type,
                "confidence": round(float(confidence), 3),
                "score": round(float(max_scores[i]), 3),
                "color": CELL_TYPE_COLORS.get(cell_type, "#cccccc")
            }
            
            # Add coordinates if available
            if barcode in barcode_to_coord:
                annotation["x"] = barcode_to_coord[barcode][0]
                annotation["y"] = barcode_to_coord[barcode][1]
            
            annotations.append(annotation)
        
        # Calculate summary statistics
        cell_type_counts = {}
        for ann in annotations:
            ct = ann["cell_type"]
            cell_type_counts[ct] = cell_type_counts.get(ct, 0) + 1
        
        # Save annotations
        annotation_result = {
            "session_id": session_id,
            "total_cells": len(annotations),
            "cell_type_counts": cell_type_counts,
            "colors": CELL_TYPE_COLORS,
            "annotations": annotations
        }
        
        annotation_path = session_dir / "cell_type_annotations.json"
        with open(annotation_path, "w") as f:
            json.dump(annotation_result, f)
        
        _annotation_status[session_id] = {
            "status": "complete",
            "message": f"Annotated {len(annotations)} cells into {len(cell_type_counts)} types",
            "cell_type_counts": cell_type_counts
        }
        
        print(f"[ANNOTATION] Session {session_id}: {len(annotations)} cells annotated")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        _annotation_status[session_id] = {
            "status": "error",
            "message": f"Annotation failed: {str(e)}"
        }

# ============================================
# Endpoints
# ============================================
@router.get("/")
def pratyaksha_root():
    """Health check for Pratyaksha module"""
    response = {
        "status": "running",
        "message": "Pratyaksha Spatial Transcriptomics Viewer Backend",
        "environment": ENV_MODE,
        "tiles_available": PRATYAKSHA_TILES_DIR.exists(),
        "tiles_path": str(PRATYAKSHA_TILES_DIR),
        "scanpy_available": SCANPY_AVAILABLE,
        "gseapy_available": GSEAPY_AVAILABLE
    }
    
    if SCANPY_AVAILABLE:
        try:
            adata = get_adata()
            response["dataset"] = {
                "barcodes": adata.n_obs,
                "genes": adata.n_vars
            }
            response["data_loaded"] = True
        except HTTPException as e:
            response["data_loaded"] = False
            response["data_message"] = e.detail
    else:
        response["data_loaded"] = False
        response["data_message"] = "Scanpy not installed - viewer works but no expression analysis"
    
    return response

@router.post("/process")
async def process_spatial_data(
    image: UploadFile = File(...),
    positions_csv: UploadFile = File(...),
    expression_h5: Optional[UploadFile] = File(None)
):
    """
    Process uploaded spatial transcriptomics data:
    1. Save uploaded files
    2. Generate Deep Zoom tiles from image
    3. Create barcodes JSON from positions CSV
    4. Optionally store expression data
    """
    session_id = str(uuid.uuid4())
    session_dir = USER_SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save uploaded image
        image_path = session_dir / f"tissue_image{Path(image.filename).suffix}"
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        print(f"[IMAGE] Saved image: {image_path} ({len(content) / 1024 / 1024:.1f} MB)")
        
        # Save and process positions CSV
        positions_path = session_dir / "tissue_positions_list.csv"
        with open(positions_path, "wb") as f:
            f.write(await positions_csv.read())
        
        # Create barcodes JSON
        barcodes_json_path = session_dir / "barcodes.json"
        try:
            # Load tissue positions - handle with or without header
            tissue_pos = pd.read_csv(positions_path, header=None)
            
            # Check if first row is a header (if first cell is "barcode")
            if str(tissue_pos.iloc[0, 0]).lower() == "barcode":
                tissue_pos = tissue_pos.iloc[1:].reset_index(drop=True)
            
            tissue_pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
            
            # Convert columns to numeric (in case they were read as strings)
            tissue_pos["in_tissue"] = pd.to_numeric(tissue_pos["in_tissue"], errors='coerce')
            tissue_pos["pxl_row_in_fullres"] = pd.to_numeric(tissue_pos["pxl_row_in_fullres"], errors='coerce')
            tissue_pos["pxl_col_in_fullres"] = pd.to_numeric(tissue_pos["pxl_col_in_fullres"], errors='coerce')
            
            # Filter: in_tissue=1 AND valid coordinates (> 0)
            tissue_pos = tissue_pos[
                (tissue_pos['in_tissue'] == 1) & 
                (tissue_pos['pxl_row_in_fullres'] > 0) & 
                (tissue_pos['pxl_col_in_fullres'] > 0) &
                tissue_pos['pxl_row_in_fullres'].notna() &
                tissue_pos['pxl_col_in_fullres'].notna()
            ]
            
            # Create barcode coordinates (using full-res pixel coordinates)
            barcode_coords = [
                {"barcode": str(row["barcode"]), "x": int(row["pxl_col_in_fullres"]), "y": int(row["pxl_row_in_fullres"])}
                for _, row in tissue_pos.iterrows()
            ]
            
            with open(barcodes_json_path, "w") as f:
                json.dump(barcode_coords, f)
            
            print(f"[SPOTS] Created barcodes JSON with {len(barcode_coords)} spots")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process positions CSV: {e}")
        
        # Generate Deep Zoom tiles
        tiles_dir = session_dir / "tiles"
        tiles_dir.mkdir(exist_ok=True)
        
        if PYVIPS_AVAILABLE:
            try:
                vips_image = pyvips.Image.new_from_file(str(image_path), access="sequential")
                dzi_path = tiles_dir / "fullres"
                vips_image.dzsave(str(dzi_path), tile_size=256, overlap=0, suffix=".jpg")
                print(f"[DZI] Generated DZI tiles at {dzi_path}")
            except Exception as e:
                print(f"[WARN] pyvips tile generation failed: {e}, using fallback")
                # Fallback: just copy the image and create a simple DZI
                create_simple_dzi(image_path, tiles_dir)
        else:
            # Simple fallback without pyvips
            create_simple_dzi(image_path, tiles_dir)
        
        # Copy barcodes to tiles directory
        shutil.copy(barcodes_json_path, tiles_dir / "barcodes_fullres.json")
        
        # Save expression data if provided
        annotation_started = False
        if expression_h5:
            h5_path = session_dir / "expression.h5"
            with open(h5_path, "wb") as f:
                f.write(await expression_h5.read())
            print(f"[DATA] Saved expression data: {h5_path}")
            
            # Start cell type annotation as background task
            if SCANPY_AVAILABLE:
                import threading
                _annotation_status[session_id] = {"status": "pending", "message": "Annotation queued..."}
                annotation_thread = threading.Thread(
                    target=run_cell_type_annotation,
                    args=(session_id,),
                    daemon=True
                )
                annotation_thread.start()
                annotation_started = True
                print(f"[ANNOTATION] Started background annotation for session {session_id}")
        
        return {
            "status": "success",
            "session_id": session_id,
            "num_spots": len(barcode_coords),
            "has_expression": expression_h5 is not None,
            "annotation_started": annotation_started,
            "tiles_path": f"/pratyaksha/session/{session_id}/tiles"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on failure
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

def create_simple_dzi(image_path: Path, tiles_dir: Path):
    """Create a DZI (Deep Zoom Image) structure without pyvips"""
    try:
        img = PILImage.open(image_path)
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        width, height = img.size
        
        tile_size = 256
        overlap = 0  # No overlap for simplicity
        
        # Calculate number of levels (Deep Zoom uses ceil(log2(max_dim)) + 1)
        max_dim = max(width, height)
        num_levels = int(np.ceil(np.log2(max_dim))) + 1
        
        # Create DZI XML
        dzi_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
  Format="jpg"
  Overlap="{overlap}"
  TileSize="{tile_size}">
  <Size Width="{width}" Height="{height}"/>
</Image>'''
        
        with open(tiles_dir / "fullres.dzi", "w") as f:
            f.write(dzi_content)
        
        # Create tiles directory structure
        files_dir = tiles_dir / "fullres_files"
        files_dir.mkdir(exist_ok=True)
        
        print(f"[DZI] Generating DZI: {width}x{height}, {num_levels} levels...")
        
        # Generate tiles for each level (level 0 = 1x1 pixel, level N = full res)
        for level in range(num_levels):
            level_dir = files_dir / str(level)
            level_dir.mkdir(exist_ok=True)
            
            # Calculate dimensions for this level
            # At level L, the image is scaled by 2^L / 2^(num_levels-1)
            scale_factor = 2 ** level / 2 ** (num_levels - 1)
            level_width = max(1, int(np.ceil(width * scale_factor)))
            level_height = max(1, int(np.ceil(height * scale_factor)))
            
            # Resize image for this level
            level_img = img.resize((level_width, level_height), PILImage.Resampling.LANCZOS)
            
            # Create tiles
            num_tiles_x = int(np.ceil(level_width / tile_size))
            num_tiles_y = int(np.ceil(level_height / tile_size))
            
            for tx in range(num_tiles_x):
                for ty in range(num_tiles_y):
                    # Calculate tile boundaries
                    x1 = tx * tile_size
                    y1 = ty * tile_size
                    x2 = min(x1 + tile_size, level_width)
                    y2 = min(y1 + tile_size, level_height)
                    
                    # Crop and save tile
                    tile = level_img.crop((x1, y1, x2, y2))
                    tile_path = level_dir / f"{tx}_{ty}.jpg"
                    tile.save(tile_path, "JPEG", quality=85)
            
            if level % 3 == 0 or level == num_levels - 1:  # Log progress
                print(f"   Level {level}: {level_width}x{level_height} ({num_tiles_x}x{num_tiles_y} tiles)")
        
        print(f"[OK] Created DZI with {num_levels} levels")
        
    except Exception as e:
        print(f"[ERROR] DZI creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

@router.get("/info")
def get_dataset_info():
    """Get dataset information"""
    adata = get_adata()
    return {
        "n_barcodes": adata.n_obs,
        "n_genes": adata.n_vars,
        "gene_names": adata.var_names.tolist()[:100],  # First 100 genes
        "barcode_sample": adata.obs_names.tolist()[:20]  # First 20 barcodes
    }

# ============================================
# Session-based Tile & Data Serving
# ============================================
@router.get("/session/{session_id}/tiles/{path:path}")
async def serve_session_tiles(session_id: str, path: str):
    """Serve tiles from a user's processing session"""
    session_dir = USER_SESSIONS_DIR / session_id / "tiles"
    file_path = session_dir / path
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Tile not found: {path}")
    
    # Determine content type
    suffix = file_path.suffix.lower()
    content_types = {
        ".dzi": "application/xml",
        ".xml": "application/xml",
        ".json": "application/json",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png"
    }
    
    return FileResponse(
        file_path,
        media_type=content_types.get(suffix, "application/octet-stream")
    )

@router.get("/session/{session_id}/barcodes")
async def get_session_barcodes(session_id: str):
    """Get barcodes JSON for a session"""
    barcodes_path = USER_SESSIONS_DIR / session_id / "tiles" / "barcodes_fullres.json"
    
    if not barcodes_path.exists():
        raise HTTPException(status_code=404, detail="Barcodes not found for this session")
    
    with open(barcodes_path) as f:
        return json.load(f)

# ============================================
# Cell Type Annotation Endpoints
# ============================================
@router.get("/session/{session_id}/annotation/status")
async def get_annotation_status(session_id: str):
    """Get the status of cell type annotation for a session"""
    if session_id in _annotation_status:
        return _annotation_status[session_id]
    
    # Check if annotation file exists (completed previously)
    annotation_path = USER_SESSIONS_DIR / session_id / "cell_type_annotations.json"
    if annotation_path.exists():
        with open(annotation_path) as f:
            data = json.load(f)
        return {
            "status": "complete",
            "message": f"Annotation complete: {data.get('total_cells', 0)} cells",
            "cell_type_counts": data.get("cell_type_counts", {})
        }
    
    # Check if expression data exists
    h5_path = USER_SESSIONS_DIR / session_id / "expression.h5"
    if not h5_path.exists():
        return {
            "status": "unavailable",
            "message": "No expression data available for annotation"
        }
    
    return {
        "status": "not_started",
        "message": "Annotation has not been started"
    }

@router.get("/session/{session_id}/annotations")
async def get_annotations(session_id: str, include_coords: bool = True):
    """Get cell type annotations for a session"""
    annotation_path = USER_SESSIONS_DIR / session_id / "cell_type_annotations.json"
    
    if not annotation_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="Annotations not found. Run annotation first or wait for it to complete."
        )
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    # Optionally strip coordinates to reduce payload size
    if not include_coords:
        for ann in data.get("annotations", []):
            ann.pop("x", None)
            ann.pop("y", None)
    
    return data

@router.post("/session/{session_id}/annotation/start")
async def start_annotation(session_id: str, background_tasks: BackgroundTasks):
    """Manually start or restart cell type annotation for a session"""
    global _annotation_status
    
    if not SCANPY_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="Scanpy not installed. Annotation requires scanpy."
        )
    
    session_dir = USER_SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    h5_path = session_dir / "expression.h5"
    if not h5_path.exists():
        raise HTTPException(
            status_code=400, 
            detail="No expression data found. Upload H5 file to enable annotation."
        )
    
    # Check if already running
    if session_id in _annotation_status and _annotation_status[session_id].get("status") == "running":
        return {
            "status": "already_running",
            "message": "Annotation is already in progress"
        }
    
    # Start annotation in background
    import threading
    _annotation_status[session_id] = {"status": "pending", "message": "Annotation queued..."}
    annotation_thread = threading.Thread(
        target=run_cell_type_annotation,
        args=(session_id,),
        daemon=True
    )
    annotation_thread.start()
    
    return {
        "status": "started",
        "message": "Cell type annotation started",
        "session_id": session_id
    }

@router.get("/annotation/markers")
async def get_available_markers():
    """Get the default marker gene sets used for annotation"""
    return {
        "markers": DEFAULT_CELL_TYPE_MARKERS,
        "colors": CELL_TYPE_COLORS
    }

@router.post("/expression")
def get_expression(request: ExpressionRequest):
    """Fetch expression values for specific barcodes and genes."""
    try:
        adata = get_adata(session_id=request.session_id)
        
        # Validate barcodes
        valid_barcodes = [bc for bc in request.barcodes if bc in adata.obs_names]
        if not valid_barcodes:
            raise HTTPException(status_code=400, detail="No valid barcodes found")

        # Handle genes - if None or contains "*", get all genes
        if request.genes is None or "*" in request.genes:
            sub = adata[valid_barcodes, :].to_df()
        else:
            valid_genes = [g for g in request.genes if g in adata.var_names]
            if not valid_genes:
                raise HTTPException(status_code=400, detail="No valid genes found")
            sub = adata[valid_barcodes, valid_genes].to_df()

        return json.loads(sub.to_json())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expression fetch failed: {e}")

@router.post("/expression_all")
def get_expression_all(request: ExpressionAllRequest):
    """Fetch expression values for ALL barcodes for given genes."""
    try:
        adata = get_adata(session_id=request.session_id)
        
        if request.genes is None or "*" in request.genes:
            # Return all genes for all barcodes (this could be very large!)
            sub = adata.to_df()
        else:
            valid_genes = [g for g in request.genes if g in adata.var_names]
            if not valid_genes:
                raise HTTPException(status_code=400, detail="No valid genes found")
            sub = adata[:, valid_genes].to_df()

        return json.loads(sub.to_json())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expression_all fetch failed: {e}")

@router.post("/dge")
def differential_expression(req: DGERequest):
    """Perform differential gene expression analysis with RC normalization."""
    try:
        adata = get_adata(session_id=req.session_id)
        
        # Validate barcodes
        g1 = [bc for bc in req.group1 if bc in adata.obs_names]
        g2 = []
        if req.group2:
            g2 = [bc for bc in req.group2 if bc in adata.obs_names]

        if len(g1) == 0:
            raise HTTPException(status_code=400, detail="No valid barcodes in group1")

        # Create a copy of adata for this analysis
        adata_copy = adata.copy()
        
        # Apply RC (Relative Counts) normalization: normalize to 10,000 counts per cell
        print("[DGE] Applying RC normalization (target_sum=10000)...")
        sc.pp.normalize_total(adata_copy, target_sum=1e4)
        sc.pp.log1p(adata_copy)

        # Case 1: group1 vs rest
        if req.vs_rest:
            adata_copy.obs["comparison"] = "rest"
            adata_copy.obs.loc[g1, "comparison"] = "group1"

            group_counts = adata_copy.obs["comparison"].value_counts()
            if group_counts.min() >= 2:
                try:
                    sc.tl.rank_genes_groups(
                        adata_copy,
                        groupby="comparison",
                        groups=["group1"],
                        reference="rest",
                        method="wilcoxon"
                    )
                    result = sc.get.rank_genes_groups_df(adata_copy, group="group1")
                    result = sanitize_results(result)
                    return result.to_dict(orient="records")
                except Exception as e:
                    print(f"Wilcoxon test failed, falling back to fold change: {e}")

            # Fallback: mean-based log2FC (data is already log-normalized)
            rest_barcodes = [bc for bc in adata_copy.obs_names if bc not in g1]
            exp1 = np.asarray(adata_copy[g1].X.mean(axis=0)).flatten()
            exp2 = np.asarray(adata_copy[rest_barcodes].X.mean(axis=0)).flatten()
            genes = adata_copy.var_names
            # For log-normalized data: log2FC = (exp1 - exp2) / log(2)
            fold_change = (exp1 - exp2) / np.log(2)
            df = pd.DataFrame({
                "names": genes, 
                "group1_mean": exp1, 
                "rest_mean": exp2, 
                "logfoldchanges": fold_change,
                "pvals": [1.0] * len(genes),
                "pvals_adj": [1.0] * len(genes)
            })
            df = sanitize_results(df)
            return df.sort_values('logfoldchanges', ascending=False).to_dict(orient="records")

        # Case 2: group1 vs group2
        if len(g2) == 0:
            raise HTTPException(status_code=400, detail="Group 2 is empty and vs_rest is False")

        # Check for overlap between groups
        g1_set = set(g1)
        g2_set = set(g2)
        overlap = g1_set & g2_set
        
        # Handle identical groups: comparing a group to itself gives fold_change=0, pval=1
        if g1_set == g2_set:
            print(f"[DGE] Groups are identical ({len(g1)} barcodes) - returning fold_change=0, pval=1")
            exp1 = np.asarray(adata_copy[g1].X.mean(axis=0)).flatten()
            genes = adata_copy.var_names
            df = pd.DataFrame({
                "names": genes,
                "group1_mean": exp1,
                "group2_mean": exp1,
                "logfoldchanges": [0.0] * len(genes),
                "pvals": [1.0] * len(genes),
                "pvals_adj": [1.0] * len(genes)
            })
            df = sanitize_results(df)
            return df.sort_values('logfoldchanges', ascending=False).to_dict(orient="records")

        # If groups overlap but aren't identical, use fold-change calculation
        # (Wilcoxon can't handle overlapping groups properly due to cell labeling)
        if len(overlap) > 0:
            print(f"[DGE] Groups have {len(overlap)} overlapping barcodes - using fold-change (no exclusion)")
            # Use ALL barcodes from each group (including overlap)
            exp1 = np.asarray(adata_copy[g1].X.mean(axis=0)).flatten()
            exp2 = np.asarray(adata_copy[g2].X.mean(axis=0)).flatten()
            genes = adata_copy.var_names
            fold_change = (exp1 - exp2) / np.log(2)
            
            df = pd.DataFrame({
                "names": genes,
                "group1_mean": exp1,
                "group2_mean": exp2,
                "logfoldchanges": fold_change,
                "pvals": [1.0] * len(genes),  # Can't compute p-values with overlap
                "pvals_adj": [1.0] * len(genes)
            })
            df = sanitize_results(df)
            return df.sort_values('logfoldchanges', ascending=False).to_dict(orient="records")

        # No overlap - can use Wilcoxon test
        # Mark cells as group1, group2, or other
        adata_copy.obs["comparison"] = "other"
        adata_copy.obs.loc[g1, "comparison"] = "group1"
        adata_copy.obs.loc[g2, "comparison"] = "group2"

        # Check if we have enough cells for statistical test
        group_counts = adata_copy.obs["comparison"].value_counts()
        
        if "group1" in group_counts and "group2" in group_counts and group_counts[["group1", "group2"]].min() >= 2:
            try:
                sc.tl.rank_genes_groups(
                    adata_copy,
                    groupby="comparison",
                    groups=["group1"],
                    reference="group2",
                    method="wilcoxon"
                )
                result = sc.get.rank_genes_groups_df(adata_copy, group="group1")
                result = sanitize_results(result)
                return result.to_dict(orient="records")
            except Exception as e:
                print(f"Wilcoxon test failed, falling back to fold change: {e}")

        # Fallback: mean-based log2FC (data is already log-normalized)
        exp1 = np.asarray(adata_copy[g1].X.mean(axis=0)).flatten()
        exp2 = np.asarray(adata_copy[g2].X.mean(axis=0)).flatten()
        genes = adata_copy.var_names
        fold_change = (exp1 - exp2) / np.log(2)
        
        df = pd.DataFrame({
            "names": genes,
            "group1_mean": exp1,
            "group2_mean": exp2,
            "logfoldchanges": fold_change,
            "pvals": [1.0] * len(genes),
            "pvals_adj": [1.0] * len(genes)
        })
        df = sanitize_results(df)
        return df.sort_values('logfoldchanges', ascending=False).to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DGE computation failed: {str(e)}")

@router.post("/go_enrichment")
def go_enrichment(request: GOEnrichmentRequest):
    """
    Perform GO enrichment analysis on a list of genes using gseapy.
    Returns JSON data grouped by ontology for frontend display.
    """
    if not GSEAPY_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="GO enrichment not available - gseapy is not installed on the server"
        )
    
    try:
        if not request.gene_list or len(request.gene_list) == 0:
            raise HTTPException(status_code=400, detail="Gene list cannot be empty")
        
        # Clean gene list
        gene_list_clean = list(set([gene.strip().upper() for gene in request.gene_list if gene.strip()]))
        
        if len(gene_list_clean) < 3:
            raise HTTPException(
                status_code=400, 
                detail="At least 3 genes required for meaningful enrichment analysis"
            )
        
        print(f"[GO] Running GO enrichment for {len(gene_list_clean)} genes...")
        
        # Map ontology names
        ontology_mapping = {
            "BP": "GO_Biological_Process_2021",
            "MF": "GO_Molecular_Function_2021",
            "CC": "GO_Cellular_Component_2021",
            "GO_Biological_Process_2021": "GO_Biological_Process_2021",
            "GO_Molecular_Function_2021": "GO_Molecular_Function_2021",
            "GO_Cellular_Component_2021": "GO_Cellular_Component_2021"
        }
        
        # Friendly names for display
        ontology_display_names = {
            "GO_Biological_Process_2021": "Biological Process",
            "GO_Molecular_Function_2021": "Molecular Function",
            "GO_Cellular_Component_2021": "Cellular Component"
        }
        
        gene_sets = [ontology_mapping.get(ont, ont) for ont in request.ontology_types]
        
        # Results grouped by ontology
        results_by_ontology = {}
        
        for gene_set in gene_sets:
            try:
                print(f"  Analyzing {gene_set}...")
                
                enr = gp.enrichr(
                    gene_list=gene_list_clean,
                    gene_sets=[gene_set],
                    organism=request.organism,
                    outdir=None,
                    cutoff=request.pvalue_threshold,
                    no_plot=True
                )
                
                if enr.results is None or enr.results.empty:
                    print(f"  No significant results for {gene_set}")
                    continue
                
                df = enr.results.copy()
                df = df[df['Adjusted P-value'] <= request.pvalue_threshold]
                df = df.sort_values('Adjusted P-value')
                
                if request.top_terms:
                    df = df.head(request.top_terms)
                
                ontology_name = ontology_display_names.get(gene_set, gene_set)
                
                terms = []
                for _, row in df.iterrows():
                    overlap_parts = row['Overlap'].split('/')
                    overlap_genes = int(overlap_parts[0])
                    overlap_total = int(overlap_parts[1])
                    adj_pval = float(row['Adjusted P-value'])
                    
                    # Compute -log10(adj_pvalue) for chart
                    neglog10_pval = -np.log10(adj_pval) if adj_pval > 0 else 0
                    
                    terms.append({
                        "term_name": row['Term'],
                        "adjusted_pvalue": adj_pval,
                        "neglog10_pvalue": round(neglog10_pval, 3),
                        "pvalue": float(row['P-value']) if 'P-value' in row else None,
                        "overlap": row['Overlap'],
                        "overlap_genes": overlap_genes,
                        "overlap_total": overlap_total,
                        "odds_ratio": round(float(row['Odds Ratio']), 3) if 'Odds Ratio' in row else None,
                        "combined_score": round(float(row['Combined Score']), 3) if 'Combined Score' in row else None,
                        "genes": row['Genes'].split(';') if isinstance(row['Genes'], str) else [],
                        "gene_count": len(row['Genes'].split(';')) if isinstance(row['Genes'], str) else 0
                    })
                
                if terms:
                    results_by_ontology[ontology_name] = terms
                
            except Exception as e:
                print(f"  Error processing {gene_set}: {str(e)}")
                continue
        
        if len(results_by_ontology) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No significant enrichment terms found. Try relaxing the p-value threshold."
            )
        
        print(f"[GO] Returning results for {len(results_by_ontology)} ontologies")
        
        return {
            "status": "success",
            "gene_count": len(gene_list_clean),
            "ontologies": results_by_ontology
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"GO enrichment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GO enrichment failed: {str(e)}")

# ============================================
# Gene Set Variation Analysis (GSVA)
# ============================================
@router.post("/gsva")
def run_gsva(request: GSVARequest):
    """
    Compute per-spot gene set variation scores for Hypoxia, EMT, and Angiogenesis.
    Uses mean z-score of genes in each set as a lightweight GSVA-like score.
    """
    try:
        adata = get_adata(session_id=request.session_id)
        
        # Determine which gene sets to score
        requested_sets = request.gene_sets or list(GSVA_GENE_SETS.keys())
        
        # Validate
        invalid = [s for s in requested_sets if s not in GSVA_GENE_SETS]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown gene sets: {invalid}. Available: {list(GSVA_GENE_SETS.keys())}"
            )
        
        # Select barcodes
        if request.barcodes and len(request.barcodes) > 0:
            valid_bc = [bc for bc in request.barcodes if bc in adata.obs_names]
            if not valid_bc:
                raise HTTPException(status_code=400, detail="No valid barcodes found")
            adata_sub = adata[valid_bc].copy()
        else:
            adata_sub = adata.copy()
        
        # Normalize (RC normalization)
        sc.pp.normalize_total(adata_sub, target_sum=1e4)
        sc.pp.log1p(adata_sub)
        
        available_genes = set(adata_sub.var_names)
        
        # Compute z-scores across all cells for proper scaling
        if hasattr(adata_sub.X, 'toarray'):
            expr_matrix = adata_sub.X.toarray()
        else:
            expr_matrix = np.array(adata_sub.X)
        
        gene_means = np.mean(expr_matrix, axis=0)
        gene_stds = np.std(expr_matrix, axis=0)
        gene_stds[gene_stds == 0] = 1  # avoid division by zero
        z_matrix = (expr_matrix - gene_means) / gene_stds
        
        results = {}
        
        for gene_set_name in requested_sets:
            genes = GSVA_GENE_SETS[gene_set_name]
            present = [g for g in genes if g in available_genes]
            
            if len(present) == 0:
                print(f"[GSVA] No genes found for {gene_set_name}, skipping")
                continue
            
            gene_indices = [adata_sub.var_names.get_loc(g) for g in present]
            
            # Mean z-score across genes in the set per cell
            set_scores = np.mean(z_matrix[:, gene_indices], axis=1)
            
            # Build per-barcode results
            scores_list = []
            for i, bc in enumerate(adata_sub.obs_names):
                scores_list.append({
                    "barcode": bc,
                    "score": round(float(set_scores[i]), 4)
                })
            
            # Statistics for color scaling
            score_values = set_scores.tolist()
            results[gene_set_name] = {
                "scores": scores_list,
                "genes_used": present,
                "genes_missing": [g for g in genes if g not in available_genes],
                "n_genes_used": len(present),
                "n_genes_total": len(genes),
                "min_score": round(float(np.min(set_scores)), 4),
                "max_score": round(float(np.max(set_scores)), 4),
                "mean_score": round(float(np.mean(set_scores)), 4),
                "colors": GSVA_COLORS.get(gene_set_name, {"low": "#2563eb", "high": "#dc2626"})
            }
            
            print(f"[GSVA] {gene_set_name}: {len(present)}/{len(genes)} genes, "
                  f"score range [{np.min(set_scores):.3f}, {np.max(set_scores):.3f}]")
        
        if len(results) == 0:
            raise HTTPException(
                status_code=400,
                detail="No gene sets could be scored - none of the genes were found in the dataset"
            )
        
        return {
            "status": "success",
            "n_barcodes": adata_sub.n_obs,
            "gene_sets": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"GSVA error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"GSVA failed: {str(e)}")

@router.get("/gsva/gene_sets")
def get_gsva_gene_sets():
    """Return available gene sets and their genes."""
    return {
        "gene_sets": {name: genes for name, genes in GSVA_GENE_SETS.items()},
        "colors": GSVA_COLORS
    }