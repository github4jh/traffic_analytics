"""
FastAPI backend for the traffic analytics dashboard.

Endpoints:
  GET  /              the dashboard (static/index.html)
  GET  /api/stats      current counters as JSON
  GET  /api/history     recent vehicles/hour snapshots, for the trend chart
  GET  /api/video       live annotated video as an MJPEG stream
  POST /api/start       (re)start processing
  POST /api/stop        stop processing
  GET  /health           liveness check for Docker/orchestrators

Run locally with:
    uvicorn app:app --reload

Run in production (see DEPLOY.md):
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from engine import TrafficEngine

traffic_engine = TrafficEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    traffic_engine.start()
    yield
    traffic_engine.stop()


app = FastAPI(
    title="Taiwan Freeway Traffic Analytics",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def dashboard():
    return FileResponse("static/index.html")


@app.get("/api/stats")
def get_stats():
    return traffic_engine.get_stats()


@app.get("/api/history")
def get_history():
    return traffic_engine.get_history()


@app.post("/api/start")
def start_engine():
    traffic_engine.start()
    return {"status": "started"}


@app.post("/api/stop")
def stop_engine():
    traffic_engine.stop()
    return {"status": "stopped"}


@app.get("/health")
def health():
    return {"status": "ok"}


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        frame = traffic_engine.get_frame_jpeg()
        if frame is not None:
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.05)


@app.get("/api/video")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
