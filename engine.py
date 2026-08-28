"""
Headless traffic-analytics engine.

Runs the same detection -> tracking -> counting -> speed pipeline as
tracker.py, but with no cv2.imshow/GUI window. Instead it keeps the
latest annotated frame (as JPEG bytes) and a snapshot of current
stats in memory, both behind a lock, so a FastAPI app can read them
concurrently from request handlers while the video loop runs in its
own background thread.
"""

import time
import threading
from collections import Counter, defaultdict, deque

import cv2
from ultralytics import YOLO, settings

# See tracker.py for why these are needed: YOLO() can otherwise make
# an unbounded network call on startup and hang.
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

import config
import analytics
import speed
import visualization as visual
from video_source import RobustVideoCapture


class TrafficEngine:
    """
    Owns one video source, one YOLO model, and all of the counting/
    speed state for that source. start()/stop() control a background
    thread; get_frame_jpeg()/get_stats()/get_history() are safe to
    call from any request handler at any time.
    """

    def __init__(self, source=None, use_live=None):
        self.use_live = (
            use_live if use_live is not None else config.USE_LIVE_SOURCE
        )
        self.source = source or (
            config.SOURCE if self.use_live else config.LOCAL
        )

        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._stats = {"running": False, "source": self.source}
        self._history = deque(maxlen=120)

        self._thread = None
        self._stop_flag = threading.Event()

        self.model = YOLO(config.MODEL_PATH)

    # -----------------------------------------------------------
    # Public controls
    # -----------------------------------------------------------

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def get_frame_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def get_stats(self):
        with self._lock:
            return dict(self._stats)

    def get_history(self):
        with self._lock:
            return list(self._history)

    # -----------------------------------------------------------
    # Background processing loop
    # -----------------------------------------------------------

    def _run(self):

        track_history = defaultdict(
            lambda: deque(maxlen=config.TRACK_HISTORY_LENGTH)
        )
        track_side = {}
        counted_ids = set()
        vehicles_passed = 0
        direction_counts = {"up": 0, "down": 0}
        speed_history = {}
        lane_counts = {
            lane: {"up": 0, "down": 0, "total": 0}
            for lane in config.LANE_RANGES
        }
        lane_counts["unassigned"] = {"up": 0, "down": 0, "total": 0}
        total_vehicle_counts = Counter()

        cap = RobustVideoCapture(
            self.source,
            reconnect=self.use_live,
            max_retries=config.RECONNECT_MAX_RETRIES,
            retry_delay=config.RECONNECT_RETRY_DELAY,
            backoff_factor=config.RECONNECT_BACKOFF_FACTOR,
            max_retry_delay=config.RECONNECT_MAX_DELAY,
        )

        if not cap.isOpened():
            with self._lock:
                self._stats = {
                    "running": False,
                    "source": self.source,
                    "error": f"Could not open source: {self.source}",
                }
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 0:
            video_fps = config.DEFAULT_FPS

        with self._lock:
            self._stats = {"running": True, "source": self.source}

        frame_id = 0
        prev_time = time.perf_counter()
        last_snapshot_time = 0.0

        while not self._stop_flag.is_set():

            success, frame = cap.read()

            if not success or frame is None:
                break

            frame_id += 1

            if frame_id % config.FRAME_SKIP != 0:
                continue

            results = self.model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                conf=config.CONFIDENCE_THRESHOLD,
                imgsz=config.IMAGE_SIZE,
                verbose=False
            )
            result = results[0]

            now = time.perf_counter()
            elapsed = now - prev_time
            processing_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            prev_time = now

            vehicle_counts = Counter()

            vehicles_passed = analytics.analyze_frame(
                result=result,
                vehicle_counts=vehicle_counts,
                track_history=track_history,
                counted_ids=counted_ids,
                vehicles_passed=vehicles_passed,
                LINE_Y=config.LINE_Y,
                direction_counts=direction_counts,
                total_vehicle_counts=total_vehicle_counts,
                lane_counts=lane_counts,
                track_side=track_side,
            )

            vehicle_speeds = {}

            if result.boxes.id is not None:

                track_ids = result.boxes.id.int().cpu().tolist()

                for track_id, box in zip(track_ids, result.boxes):

                    class_id = int(box.cls[0])
                    if class_id not in config.VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_y = int(y2)

                    speed_kmh = speed.update_speed(
                        track_id=track_id,
                        center_y=center_y,
                        frame_id=frame_id,
                        video_fps=video_fps,
                        speed_history=speed_history,
                        line1_y=config.SPEED_LINE_1_Y,
                        line2_y=config.SPEED_LINE_2_Y,
                        distance_m=config.SPEED_LINE_DISTANCE_M
                    )

                    if speed_kmh is not None:
                        vehicle_speeds[track_id] = speed_kmh

            video_time = frame_id / video_fps
            vehicles_per_hour = (
                vehicles_passed / video_time * 3600
                if video_time > 0 else 0
            )
            current_vehicle_count = sum(vehicle_counts.values())

            annotated_frame = visual.display_output(
                result=result,
                current_vehicle_count=current_vehicle_count,
                vehicle_counts=vehicle_counts,
                track_history=track_history,
                LINE_Y=config.LINE_Y,
                vehicles_passed=vehicles_passed,
                processing_fps=processing_fps,
                direction_counts=direction_counts,
                total_vehicle_counts=total_vehicle_counts,
                vehicles_per_hour=vehicles_per_hour,
                vehicle_speeds=vehicle_speeds,
                lane_counts=lane_counts,
                lane_ranges=config.LANE_RANGES,
            )

            ok, buf = cv2.imencode(
                ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            jpeg_bytes = buf.tobytes() if ok else None

            stats_snapshot = {
                "running": True,
                "source": self.source,
                "timestamp": time.time(),
                "processing_fps": round(processing_fps, 1),
                "current_vehicle_count": current_vehicle_count,
                "vehicle_counts": dict(vehicle_counts),
                "vehicles_passed": vehicles_passed,
                "direction_counts": dict(direction_counts),
                "total_vehicle_counts": dict(total_vehicle_counts),
                "lane_counts": {k: dict(v) for k, v in lane_counts.items()},
                "vehicles_per_hour": round(vehicles_per_hour, 1),
            }

            with self._lock:
                if jpeg_bytes is not None:
                    self._latest_jpeg = jpeg_bytes
                self._stats = stats_snapshot

                if now - last_snapshot_time >= 1.0:
                    last_snapshot_time = now
                    self._history.append({
                        "timestamp": stats_snapshot["timestamp"],
                        "vehicles_per_hour": stats_snapshot["vehicles_per_hour"],
                        "vehicles_passed": stats_snapshot["vehicles_passed"],
                    })

        cap.release()

        with self._lock:
            self._stats["running"] = False
