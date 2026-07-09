FROM python:3.11-slim

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first to prevent pyannote from pulling torch+CUDA (several GB)
# and to pin a version compatible with pyannote.audio 3.3.2
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

# Respect $PORT if the platform injects one (e.g. Railway), else default to 8000
# (docker-compose / local)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
