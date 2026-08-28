import cv2
import time
import speed

from collections import Counter
from collections import defaultdict, deque

from ultralytics import YOLO, settings

# Ultralytics can make a network call on startup (PyPI version check /
# HUB analytics) with no timeout. On a slow, filtered, or VPN'd
# connection this is what causes tracker.py to sit with zero output
# for minutes -- settings.update(sync=False) alone doesn't reliably
# stop it on every version, so the ONLINE flag is forced off directly
# as well, which is what actually short-circuits the check.
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
import visualization as visual
from video_source import RobustVideoCapture


# ---------------------------------------
# Tracking history
# ---------------------------------------

track_history = defaultdict(
    lambda: deque(
        maxlen=config.TRACK_HISTORY_LENGTH
    )
)

# Which side of LINE_Y each track currently sits on ("above" /
# "below"). Used by analytics.detect_crossing for hysteresis-based
# crossing detection (bug 1 fix).
track_side = {}

counted_ids = set()

vehicles_passed = 0

direction_counts = {
    "up": 0,
    "down": 0
}

# ---------------------------------------
# Speed history
# ---------------------------------------
speed_history = {}

lane_counts = {
    lane: {
        "up": 0,
        "down": 0,
        "total": 0
    }
    for lane in config.LANE_RANGES
}

# Vehicles that cross the line outside every configured lane range
# land here instead of being silently dropped (bug 2 fix support).
lane_counts["unassigned"] = {
    "up": 0,
    "down": 0,
    "total": 0
}

# ---------------------------------------
# Load YOLO model
# ---------------------------------------

model = YOLO(config.MODEL_PATH)


# ---------------------------------------
# Open video source (local file, or live CCTV stream with
# auto-reconnect -- bug 4 fix)
# ---------------------------------------

source = config.SOURCE if config.USE_LIVE_SOURCE else config.LOCAL

cap = RobustVideoCapture(
    source,
    # Only retry forever for the live CCTV feed. config.LOCAL is a
    # finite file -- a failed read there just means end-of-video,
    # and reconnect logic would turn that into an infinite loop.
    reconnect=config.USE_LIVE_SOURCE,
    max_retries=config.RECONNECT_MAX_RETRIES,
    retry_delay=config.RECONNECT_RETRY_DELAY,
    backoff_factor=config.RECONNECT_BACKOFF_FACTOR,
    max_retry_delay=config.RECONNECT_MAX_DELAY,
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video source: {source}"
    )


# ---------------------------------------
# Video information
# ---------------------------------------

video_fps = cap.get(cv2.CAP_PROP_FPS)

if not video_fps or video_fps <= 0:
    print(
        f"[warn] Source reported FPS={video_fps}, falling back to "
        f"config.DEFAULT_FPS={config.DEFAULT_FPS}. Speed and "
        f"vehicles/hour figures will only be as accurate as this "
        f"value."
    )
    video_fps = config.DEFAULT_FPS

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

# Live MJPEG streams often report 0 for width/height/frame_count
# before any frame has actually arrived. Pull one frame up front to
# get real dimensions in that case.
probe_frame = None

if width <= 0 or height <= 0:

    success, probe_frame = cap.read()

    if not success or probe_frame is None:
        raise RuntimeError(
            f"Could not read a frame from source: {source}"
        )

    height, width = probe_frame.shape[:2]

duration = (
    frame_count / video_fps
    if (video_fps > 0 and frame_count > 0)
    else 0
)

print("--------------------------------")
print("Video information")
print("--------------------------------")

print(f"Source     : {source}")
print(f"Resolution : {width} x {height}")
print(f"FPS        : {video_fps:.2f}")
print(
    f"Frames     : "
    f"{frame_count if frame_count > 0 else 'unknown (live stream)'}"
)
print(f"Duration   : {duration:.2f} seconds")


# ---------------------------------------
# Frame skipping
# ---------------------------------------

FRAME_SKIP = config.FRAME_SKIP

output_fps = video_fps / FRAME_SKIP

print(f"Frame skip : {FRAME_SKIP}")
print(f"Output FPS : {output_fps:.2f}")


# ---------------------------------------
# Video writer
# ---------------------------------------

writer = cv2.VideoWriter(
    config.OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    output_fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        f"Could not create output video: "
        f"{config.OUTPUT}"
    )


# ---------------------------------------
# Processing
# ---------------------------------------

frame_id = 0

prev_time = time.perf_counter()

total_vehicle_counts = Counter()

while True:

    # -----------------------------------
    # Read frame
    # -----------------------------------

    if probe_frame is not None:
        # Reuse the frame we already grabbed while probing
        # dimensions above, so it doesn't just get thrown away.
        success, frame = True, probe_frame
        probe_frame = None
    else:
        success, frame = cap.read()

    if not success or frame is None:

        print("End of video / stream unavailable.")

        break

    frame_id += 1


    # -----------------------------------
    # Skip frames
    # -----------------------------------

    if frame_id % FRAME_SKIP != 0:
        continue

    # -----------------------------------
    # Run YOLO + BoT-SORT
    # -----------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=config.CONFIDENCE_THRESHOLD,
        imgsz=config.IMAGE_SIZE,
        verbose=False
    )

    result = results[0]

    # -----------------------------------
    # Calculate processing FPS
    # -----------------------------------

    current_time = time.perf_counter()

    elapsed = current_time - prev_time

    if elapsed > 0:
        processing_fps = 1.0 / elapsed
    else:
        processing_fps = 0.0

    prev_time = current_time


    # -----------------------------------
    # Analyze vehicles
    # -----------------------------------
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
        track_side=track_side
    )

    # ---------------------------------------
    # Speed estimation
    # ---------------------------------------

    vehicle_speeds = {}

    if result.boxes.id is not None:

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        for track_id, box in zip(
            track_ids,
            result.boxes
        ):

            class_id = int(
                box.cls[0]
            )

            if class_id not in config.VEHICLE_CLASSES:
                continue


            # Use bottom-center of bounding box
            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

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

    if video_time > 0:
        vehicles_per_hour = (
            vehicles_passed / video_time * 3600
        )
    else:
        vehicles_per_hour = 0
    # -----------------------------------
    # Current vehicle count
    # -----------------------------------

    current_vehicle_count = sum(
        vehicle_counts.values()
    )


    # -----------------------------------
    # Visualization
    # -----------------------------------

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
        lane_ranges=config.LANE_RANGES
    )


    # -----------------------------------
    # Write output
    # -----------------------------------

    writer.write(annotated_frame)


    # -----------------------------------
    # Display
    # -----------------------------------

    cv2.imshow(
        "YOLO Traffic Analytics",
        annotated_frame
    )


    # -----------------------------------
    # Quit
    # -----------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ---------------------------------------
# Cleanup
# ---------------------------------------

writer.release()

cap.release()

cv2.destroyAllWindows()

print("--------------------------------")
print("Processing complete")
print(f"Output: {config.OUTPUT}")
print("--------------------------------")
