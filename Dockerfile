# ── CPU-only image (no CUDA needed for dev) ──────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps for soundfile / audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installs
RUN pip install --no-cache-dir uv

# Install LavaSR from YOUR fork (contains the repo_type fix)
RUN uv pip install --system --no-cache \
        git+https://github.com/memoreco-dev/LavaSR.git

# Install RunPod SDK and soundfile
RUN uv pip install --system --no-cache \
        runpod \
        soundfile

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
