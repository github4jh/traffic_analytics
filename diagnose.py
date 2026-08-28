import time

t0 = time.perf_counter()
print(f"[{time.perf_counter()-t0:.1f}s] starting imports...", flush=True)

import cv2
print(f"[{time.perf_counter()-t0:.1f}s] cv2 imported (version {cv2.__version__})", flush=True)

import torch
print(f"[{time.perf_counter()-t0:.1f}s] torch imported (version {torch.__version__})", flush=True)
print(f"    CUDA available : {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"    GPU            : {torch.cuda.get_device_name(0)}", flush=True)
print(f"    CPU threads    : {torch.get_num_threads()}", flush=True)

from ultralytics import YOLO, settings
print(f"[{time.perf_counter()-t0:.1f}s] ultralytics imported", flush=True)

settings.update({"sync": False})
try:
    import ultralytics.utils as _ultra_utils
    _ultra_utils.ONLINE = False
except Exception:
    pass
try:
    import ultralytics.hub.utils as _hub_utils
    _hub_utils.ONLINE = False
except Exception:
    pass

print(f"[{time.perf_counter()-t0:.1f}s] loading model...", flush=True)
model = YOLO("yolo26n.pt")
print(f"[{time.perf_counter()-t0:.1f}s] model loaded", flush=True)
print(f"    model.device = {model.device}", flush=True)

cap = cv2.VideoCapture("videos/highway_traffic.mp4")
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read a frame from videos/highway_traffic.mp4")

# Run inference several times on the SAME frame to separate one-time
# cold-start cost (JIT/graph compilation, thread pool warmup, cudnn
# autotune) from the steady-state per-frame speed.
for i in range(5):
    t_start = time.perf_counter()
    results = model.track(
        frame, persist=True, tracker="botsort.yaml",
        conf=0.5, imgsz=320, verbose=False
    )
    t_elapsed = time.perf_counter() - t_start
    print(f"[{time.perf_counter()-t0:.1f}s] inference #{i+1}: {t_elapsed*1000:.1f} ms, {len(results[0].boxes)} boxes", flush=True)

print(f"[{time.perf_counter()-t0:.1f}s] ALL DONE", flush=True)
