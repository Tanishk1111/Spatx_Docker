FROM python:3.11-slim-bookworm

# ── System libraries ──────────────────────────────────────────
# libvips       : fast DZI tile generation (Pratyaksha)
# libgl1 etc.   : OpenCV / torchvision transitive deps
# libjpeg/png/tiff + libhdf5 : image & HDF5 codec support
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    for i in 1 2 3; do \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            libvips42 libvips-dev \
            build-essential pkg-config \
            libgl1 libglib2.0-0 \
            libjpeg62-turbo libpng16-16 libtiff6 \
            libhdf5-dev \
        && break || { echo "Attempt $i failed, retrying..."; sleep 5; }; \
    done

WORKDIR /app

# ── Python dependencies (cached unless requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source code ──────────────────────────────────
COPY src/ /app/

# ── Install spatx_core as proper Python package ──────────────
# This makes `from spatx_core.spatx_core.models...` work (directory based)
# AND keeps the package installable if needed later.
# No pip install needed — the directory structure handles imports.

# ── Database initialiser + entrypoint ────────────────────────
COPY init_db.py /app/init_db.py
COPY entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# ── Runtime directories ──────────────────────────────────────
RUN mkdir -p /app/data /app/uploads /app/pratyaksha_data \
             /app/pratyaksha_tiles /app/pratyaksha_sessions \
             /app/user_models /app/training_data

EXPOSE 9001

ENTRYPOINT ["/app/entrypoint.sh"]
