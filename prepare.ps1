# ============================================================
# SpatX Docker Build Preparation Script
# Run this ONCE before `docker compose up --build`
# ============================================================
$ErrorActionPreference = "Stop"
$ROOT = Split-Path $PSScriptRoot -Parent  # parent = BTP project root

Write-Host "=== Preparing SpatX Docker Build ===" -ForegroundColor Cyan
Write-Host "Source: $ROOT"

# --- Backend ---
$BE = "$PSScriptRoot\backend\src"
New-Item -ItemType Directory -Force -Path $BE | Out-Null

# Core application files
$appFiles = @(
    "app_enhanced.py",
    "app_simple.py",
    "app_pratyaksha.py",
    "app_training.py",
    "database.py",
    "models.py",
    "gene_metadata.py"
)
foreach ($f in $appFiles) {
    Copy-Item "$ROOT\$f" "$BE\$f" -Force
    Write-Host "  Copied $f" -ForegroundColor Green
}

# spatx_core package (entire tree)
if (Test-Path "$BE\spatx_core") { Remove-Item "$BE\spatx_core" -Recurse -Force }
Copy-Item "$ROOT\spatx_core" "$BE\spatx_core" -Recurse -Force
Write-Host "  Copied spatx_core/" -ForegroundColor Green

# saved_models directory
if (Test-Path "$BE\saved_models") { Remove-Item "$BE\saved_models" -Recurse -Force }
Copy-Item "$ROOT\saved_models" "$BE\saved_models" -Recurse -Force
Write-Host "  Copied saved_models/" -ForegroundColor Green

# deploy/init_database.py
New-Item -ItemType Directory -Force -Path "$BE\deploy" | Out-Null
Copy-Item "$ROOT\deploy\init_database.py" "$BE\deploy\init_database.py" -Force
Write-Host "  Copied deploy/init_database.py" -ForegroundColor Green

# Create empty runtime directories (Docker will use these or mount volumes)
foreach ($dir in @("uploads", "pratyaksha_data", "pratyaksha_tiles", "pratyaksha_sessions", "data", "user_models", "training_data")) {
    New-Item -ItemType Directory -Force -Path "$BE\$dir" | Out-Null
}

# --- Frontend ---
$FE = "$PSScriptRoot\frontend\src"
New-Item -ItemType Directory -Force -Path $FE | Out-Null
Copy-Item "$ROOT\frontend_working\index.html" "$FE\index.html" -Force
Write-Host "  Copied frontend_working/index.html" -ForegroundColor Green

Write-Host ""
Write-Host "=== Build context ready! ===" -ForegroundColor Cyan
Write-Host "Next steps:"
Write-Host "  cd docker_deploy"
Write-Host "  docker compose up --build"
Write-Host ""
Write-Host "Default login:  admin / admin123" -ForegroundColor Yellow
