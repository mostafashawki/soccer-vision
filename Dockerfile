FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.yaml .
COPY main.py .
COPY pipeline/ pipeline/
COPY api/ api/
COPY ui/ ui/
COPY utils/ utils/
COPY scripts/ scripts/

# Download YOLO weights at build time (cached in image)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Default to API server
EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
