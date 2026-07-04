# ── Local AI Structify Docker Image ────────────────────────────────────
# Multi-stage build: install system deps, then Python dependencies,
# then run the application.
# ────────────────────────────────────────────────────────────────────────

# Stage 1: Base — system dependencies
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies required by the application:
#   - tesseract-ocr       : OCR extraction
#   - ffmpeg              : audio/video processing
#   - poppler-utils       : PDF rendering (pdftoppm)
#   - libgl1, libglib2.0  : OpenCV dependencies
#   - build-essential, git: Python package compilation
#   - curl                : health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        ffmpeg \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
        build-essential \
        git \
        curl \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app


# Stage 2: Python dependencies
FROM base AS dependencies

COPY requirements.txt .

# Install all Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Stage 3: Runtime image
FROM base AS runtime

# Copy Python packages from the dependencies stage
COPY --from=dependencies /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=dependencies /usr/local/bin/ /usr/local/bin/

# Copy application source code
COPY . .

# Create persistent data directories (these will be overlaid by volumes at runtime)
RUN mkdir -p /app/uploads /app/cache /app/data /app/models /app/logs

# Expose Streamlit's default port
EXPOSE 8501

# Health check: verify the application process is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: run the application launcher
# python app.py runs health checks then launches Streamlit on 0.0.0.0:8501
# (configured via STREAMLIT_SERVER_ADDRESS and STREAMLIT_SERVER_PORT env vars)
CMD ["python", "app.py"]
