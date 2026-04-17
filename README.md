# SpatX Docker

Dockerised deployment of the **SpatX** spatial transcriptomics platform — a FastAPI + PyTorch backend with an nginx frontend, all orchestrated by Docker Compose.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine + Compose (Linux)
- ~8 GB free disk space (PyTorch CPU images are large)

### Recommended: fix Docker DNS (Windows)

If you hit slow downloads or timeouts during the build, open **Docker Desktop → Settings → Docker Engine** and add:

```json
{
  "dns": ["8.8.8.8", "8.8.4.4"]
}
```

Click **Apply & Restart**.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Tanishk1111/Spatx_Docker.git
cd Spatx_Docker

# 2. Build & start (first run takes ~15-20 min for PyTorch)
docker compose up --build

# 3. Open in browser
#    Frontend:  http://localhost:8080
#    API docs:  http://localhost:9001/docs
```

A default admin account is created automatically on first run:

| Field    | Value      |
|----------|------------|
| Username | `admin`    |
| Password | `admin123` |

## Stopping & Restarting

```bash
# Stop (preserves database & uploads)
docker compose down

# Restart later (no rebuild needed)
docker compose up
```

## Features

### Gene Expression Prediction

Predict spatial gene expression from histology images using a pre-trained CiT-Net deep learning model.

**Workflow:**

1. **Upload** a tissue image (`.tif`, `.tiff`, `.png`, `.jpg`)
2. **Choose a model** -- the bundled 50-gene breast cancer model, or your own fine-tuned model
3. **Select prediction density** -- controls the spacing between prediction points (low / medium / high / full)
4. **Pick genes** from the 50-gene panel (or select all)
5. **View results** -- per-gene heatmaps, overlays, and contour maps rendered on top of the original tissue image, with expression statistics

The backend extracts 224x224 patches at each grid point, runs inference through the CiT-Net model, and generates per-gene visualisations (heatmap, overlay, contour PNGs).

### Model Training

Fine-tune the prediction model on your own spatial transcriptomics data.

**Workflow:**

1. **Upload** a tissue image + CSV containing `x`, `y` coordinates and gene expression columns
2. **Configure** training parameters (epochs, learning rate, batch size, validation split)
3. **Train** -- the CiT-Net backbone is frozen and only the regression head is fine-tuned
4. **Monitor** real-time progress via a polling-based progress bar
5. Your trained model is saved and can be selected for future predictions

### Spatial Viewer (Pratyaksha)

Interactive exploration of spatial transcriptomics datasets (10x Visium-style).

**Workflow:**

1. **Upload** a tissue image, `tissue_positions_list.csv`, and optionally a 10x `.h5` expression matrix
2. **Explore** the tissue at full resolution via a deep-zoom (DZI) viewer powered by OpenSeadragon
3. **Draw ROIs** (polygon regions of interest) on the tissue to select spot subsets
4. **Query gene expression** for selected spots or the full dataset
5. **Run analyses** on the selected regions:
   - **Differential Gene Expression (DGE)** -- Wilcoxon rank-sum between ROI groups or group-vs-rest
   - **GO Enrichment** -- Gene Ontology enrichment analysis via Enrichr (Biological Process, Molecular Function, Cellular Component)
   - **GSVA Scoring** -- Pathway-level scores (Hypoxia, EMT, Angiogenesis) visualised as spot overlays
6. **Export** results as JSON/CSV, or save/import full session state

**Accepted file formats:**

| File | Format |
|------|--------|
| Tissue image | `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg` |
| Spot positions | `tissue_positions_list.csv` (Visium `spatial/` format) |
| Expression matrix | 10x Genomics filtered feature barcode `.h5` |
| Coordinates CSV | `x`, `y` columns (auto-generated for predictions) |

## Architecture

| Service    | Image              | Port  | Description                           |
|------------|--------------------|-------|---------------------------------------|
| `backend`  | python:3.11-slim   | 9001  | FastAPI + PyTorch (CPU) ML backend    |
| `frontend` | nginx:1.27-alpine  | 8080  | SPA served by nginx, proxies API calls|

Data is persisted in Docker volumes (`db-data`, `upload-data`, `session-data`) so it survives container restarts.

## Rebuilding After Code Changes

```bash
# Rebuild only the changed service
docker compose up --build backend
docker compose up --build frontend

# Full clean rebuild
docker compose down
docker compose build --no-cache
docker compose up
```
