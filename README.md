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
