import os

MODEL_PATH = "yolo26n.pt"

CONFIDENCE_THRESHOLD = 0.5

IMAGE_SIZE = 320

FRAME_SKIP = 1

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

LOCAL = "videos/highway_traffic.mp4"

OUTPUT = "output/traffic_result.mp4"

SOURCE = "https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=19800"

# ---------------------------------------
# Source selection
# ---------------------------------------

# Set True to pull frames from the live Taiwan Freeway CCTV feed
# (SOURCE) instead of the local test video (LOCAL). Overridable via
# the USE_LIVE_SOURCE environment variable (set by the Dockerfile),
# so the same image can run against the live feed in production or
# the bundled test clip locally without a code change.
USE_LIVE_SOURCE = os.environ.get(
    "USE_LIVE_SOURCE", "false"
).strip().lower() in ("1", "true", "yes")

# Fallback FPS used when the source does not report a usable one.
# Live MJPEG/HTTP CCTV streams frequently report 0 for CAP_PROP_FPS,
# which used to silently break the speed/flow-rate math.
DEFAULT_FPS = 25.0

# ---------------------------------------
# Stream reconnection (bug 4)
# ---------------------------------------

# None = retry forever. Set an integer to give up after N attempts.
RECONNECT_MAX_RETRIES = None
RECONNECT_RETRY_DELAY = 2.0        # seconds, initial delay
RECONNECT_BACKOFF_FACTOR = 1.5     # delay grows after each failed attempt
RECONNECT_MAX_DELAY = 30.0         # seconds, cap on retry delay

# ---------------------------------------
# Traffic counting line
# ---------------------------------------

# Measured for the local test video (highway_traffic.mp4, 1857x666).
LINE_Y_LOCAL = 450

# Measured for the live camera 19800 feed (349x235) via
# calibrate_lanes_for_live_highway_traffic.py -- same relative
# position (67.6% down the frame) as LINE_Y_LOCAL, since the two
# sources have very different resolutions.
LINE_Y_LIVE = 159

LINE_Y = LINE_Y_LIVE if USE_LIVE_SOURCE else LINE_Y_LOCAL

# Half-width (pixels) of the hysteresis band around LINE_Y. A vehicle
# must clear this band on the far side before it is counted as having
# crossed. It needs to be non-zero (so a vehicle sitting exactly on
# the line doesn't flicker) but the detection itself no longer
# depends on the vehicle jumping across the whole band in one frame
# (see analytics.detect_crossing).
# NOTE: tuned for the local video's resolution. The live feed's frame
# is ~5x smaller in each dimension, so this margin may end up
# proportionally too large there -- watch for undercounting on the
# live source and shrink this if crossings feel sluggish to register.
LINE_MARGIN = 6

# ---------------------------------------
# Speed estimation
# ---------------------------------------

# NOTE: these two lines (and SPEED_LINE_DISTANCE_M) are only
# calibrated for the local test video. They have NOT been calibrated
# for the live camera 19800 feed -- treat any speed output as
# meaningless when USE_LIVE_SOURCE=True until these get a live-source
# counterpart the same way LINE_Y/LANE_RANGES did.
SPEED_LINE_1_Y = 400
SPEED_LINE_2_Y = 500

# Real-world distance between the two speed measurement lines.
# YOU MUST calibrate this value for your video.
SPEED_LINE_DISTANCE_M = 35.0

# ---------------------------------------
# Trajectory
# ---------------------------------------

TRACK_HISTORY_LENGTH = 40

# ---------------------------------------
# Lane configuration
# ---------------------------------------

# NOTE: these are pixel x-ranges measured at the row LINE_Y, and are
# specific to how each source is framed -- a set calibrated for one
# source's resolution/camera angle will NOT be correct for the
# other. Run calibrate_lanes.py (local video) or
# calibrate_lanes_for_live_highway_traffic.py (live camera 19800) and
# inspect the resulting output/*.png, then adjust these ranges until
# the drawn boundaries line up with the real lane markings before
# trusting the per-lane counts.

# highway_traffic.mp4 -- 6 lanes, no direction split in the naming
# (the test clip only shows traffic moving one direction).
LANE_RANGES_LOCAL = {
    "lane_1": (400, 520),
    "lane_2": (520, 620),
    "lane_3": (620, 760),
    "lane_4": (790, 1080),
    "lane_5": (1080, 1180),
    "lane_6": (1180, 1320),
}

# Live camera 19800 (Freeway 1 Northbound, 98K+000, Hsinchu section)
# -- 7 lanes: left 3 southbound/"down" (far carriageway, moving away
# from the camera), right 4 northbound/"up" (near carriageway, moving
# toward the camera), split by the median.
LANE_RANGES_LIVE = {
    "lane_down_1": (52, 72),
    "lane_down_2": (72, 92),
    "lane_down_3": (92, 112),
    "lane_up_1": (140, 164),
    "lane_up_2": (164, 189),
    "lane_up_3": (189, 214),
    "lane_up_4": (214, 239),
}

LANE_RANGES = LANE_RANGES_LIVE if USE_LIVE_SOURCE else LANE_RANGES_LOCAL

# Derived from the selected dict above rather than hardcoded, so it
# can never drift out of sync with LANE_RANGES again.
LANE_COUNT = len(LANE_RANGES)
