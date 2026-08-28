---
title: Freeway Traffic Analytics
emoji: 🚗
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# Taiwan Freeway Traffic Analytics

Real-time vehicle detection, tracking, counting, and speed estimation
on Taiwan Freeway CCTV footage, built with YOLO + BoT-SORT/ByteTrack,
served through a FastAPI backend and a live web dashboard.

Pipeline: CCTV / video stream -> YOLO object detection ->
BoT-SORT/ByteTrack object tracking -> traffic counting (entry, exit,
occupancy, per-lane) + vehicle analytics (type, track duration,
density) -> FastAPI backend -> web dashboard.

## Run locally

```
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
uvicorn app:app --reload
```

Open http://localhost:8000.

## Run with Docker

```
docker compose up --build
```

See `DEPLOY.md` for cloud deployment notes, and
`calibrate_lanes.py` / `calibrate_lanes_for_live_highway_traffic.py`
if you're pointing this at a different video or camera and need to
re-tune `LANE_RANGES`/`LINE_Y` in `config.py`.

The YAML block at the top of this file is Hugging Face Spaces
configuration (tells it to build this repo's `Dockerfile` and route
traffic to port 8000) -- GitHub just ignores it and renders the rest
of this file normally.
