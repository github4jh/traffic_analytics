# Deployment

## Run locally (no Docker)

```
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build if you have a GPU
uvicorn app:app --reload
```

Open http://localhost:8000. By default `USE_LIVE_SOURCE` is unset,
so it processes `videos/highway_traffic.mp4`. Set the env var to
switch to the live CCTV feed:

```
# Windows PowerShell
$env:USE_LIVE_SOURCE = "true"; uvicorn app:app --reload

# macOS/Linux
USE_LIVE_SOURCE=true uvicorn app:app --reload
```

## Run with Docker

```
docker build -t traffic-analytics .
docker run -p 8000:8000 traffic-analytics
```

or with Compose:

```
docker compose up --build
```

The image defaults to `USE_LIVE_SOURCE=true` (see `Dockerfile`),
since there's no reason to ship a container that just replays a
bundled test clip in production. Override at run time if you want
the test video instead: `docker run -p 8000:8000 -e USE_LIVE_SOURCE=false traffic-analytics`.

## Deploying to the cloud

This service is a **single stateful process**: one thread holds the
YOLO model, the tracker state, and the in-memory latest frame/stats.
That has real implications for where it fits:

- **Good fit:** any platform that runs one long-lived container and
  gives it a stable public URL -- a plain VM (AWS EC2, GCP Compute
  Engine, Azure VM, or a basic DigitalOcean/Linode droplet) with
  Docker installed, or a "always-on" PaaS container service like
  Render, Fly.io, or Railway.
- **Poor fit:** platforms that scale to zero or run multiple
  replicas behind a load balancer by default (e.g. Cloud Run's
  default autoscaling, Fargate with >1 task). Each replica would
  independently open its own connection to the CCTV source and run
  its own detector, which is wasteful and means `/api/stats` would
  return different numbers depending on which replica answers. If
  you do use such a platform, pin it to exactly one instance/replica
  and disable scale-to-zero (a paused instance can't process video).

### Minimal VM deployment

1. Provision a VM (2 vCPU / 4 GB RAM is enough for the nano model at
   `imgsz=320`) and install Docker.
2. Copy the project over (`git clone` or `scp`), then:
   ```
   docker compose up -d --build
   ```
3. Open the firewall for port 8000 (or put a reverse proxy like
   Caddy/nginx in front on port 443 with a real TLS cert -- the
   MJPEG stream and JSON endpoints both work fine behind a normal
   HTTP reverse proxy).
4. Visit `http://<vm-ip>:8000` (or your domain).

### Things worth tightening before this is public-facing

- `/api/start` and `/api/stop` are unauthenticated -- add an API key
  check or put them behind auth before exposing this beyond your own
  network.
- CORS isn't configured, since the dashboard is served from the same
  origin as the API. If you split the frontend out later, add
  `fastapi.middleware.cors.CORSMiddleware`.
- The MJPEG stream re-encodes every frame to JPEG on the server;
  fine for one or a handful of viewers, but it won't scale to many
  concurrent dashboard viewers without a proper media server (e.g.
  re-streaming via HLS) in front.
