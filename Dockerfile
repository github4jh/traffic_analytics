FROM python:3.11-slim

# libgl1/libglib2.0-0 are required by opencv even in headless mode
# (it still links against them); ffmpeg backs cv2's video decoding;
# curl is only used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CPU-only torch wheel first, in its own layer -- much smaller than
# the default CUDA build and rarely changes, so it stays cached
# across rebuilds of your actual code.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Points the pipeline at the live CCTV feed instead of the bundled
# test video by default -- override with `-e USE_LIVE_SOURCE=false`
# to use videos/highway_traffic.mp4 inside the container instead.
ENV USE_LIVE_SOURCE=true
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
