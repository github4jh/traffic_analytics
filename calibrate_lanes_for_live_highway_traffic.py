"""
Lane calibration helper for the live camera 19800 feed
(https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=19800 -- Freeway
1 Northbound, 98K+000, Hsinchu section).

Same idea as calibrate_lanes.py, but pointed at this live camera
instead of the local test video, and seeded with a starting guess
that matches this camera's layout: 7 lanes total, viewed from an
overhead gantry looking along the road, split by the median into

    left 3 lanes  -> southbound / "down"  (far carriageway, moving
                     away from the camera toward the top of frame)
    right 4 lanes -> northbound / "up"    (near carriageway, moving
                     toward the camera)

These x-ranges are still just a guess based on the screenshot --
the whole point of this script is to draw them so you can compare
against the real lane markings and adjust before trusting any
per-lane counts. LINE_Y is derived from this camera's actual frame
height (see LINE_Y_FRAC below), not hardcoded, since this feed's
resolution (349x235 observed) is much smaller than the original test
video's, and a literal LINE_Y=450 falls off the bottom of the frame
entirely.

Usage:
    python calibrate_lanes_for_live_highway_traffic.py
    -> writes output/lane_calibration_live_19800.png
    -> prints a LANE_RANGES dict ready to paste into config.py

If cv2.VideoCapture can't read the URL directly (some of these
gov.tw endpoints serve a single refreshing JPEG rather than a true
MJPEG multipart stream, which OpenCV's FFmpeg backend doesn't always
handle), this falls back to a plain HTTP GET + cv2.imdecode.
"""

import os
import time
import urllib.request

import cv2
import numpy as np

CAMERA_URL = "https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=19800"
OUTPUT_PATH = "output/lane_calibration_live_19800.png"

# LINE_Y=450 was tuned for the original test video, which is
# 1857x666 -- that's 450/666 = 67.6% of the way down the frame. This
# camera's feed turned out to be only 349x235, so a literal
# LINE_Y=450 falls 215px below the bottom edge and never gets drawn.
# Rather than hardcode a second pixel value that will break again
# the next time the resolution changes, compute LINE_Y as that same
# fraction of whatever height this camera actually returns.
REFERENCE_HEIGHT = 666
REFERENCE_LINE_Y = 450
LINE_Y_FRAC = REFERENCE_LINE_Y / REFERENCE_HEIGHT  # 0.676

# How many attempts to grab a usable frame before giving up -- these
# feeds occasionally hiccup on a single request.
MAX_ATTEMPTS = 5
RETRY_DELAY = 2.0

# ---------------------------------------------------------------
# Starting guess for the lane layout, as fractions of frame width.
# Adjust these fractions (not raw pixels) and rerun until the drawn
# lines line up -- fractions stay correct even if the feed's actual
# resolution turns out different from what you expect.
# ---------------------------------------------------------------
DOWN_LANES = 3   # left group, far carriageway
UP_LANES = 4     # right group, near carriageway

LEFT_MARGIN_FRAC = 0.15     # left edge of frame -> start of lane_down_1
DOWN_UP_GAP_START_FRAC = 0.32   # end of lane_down_3 / start of median
DOWN_UP_GAP_END_FRAC = 0.40     # end of median / start of lane_up_1
RIGHT_MARGIN_FRAC = 0.68    # end of lane_up_4 -> right edge of frame


def grab_frame():
    """Try cv2.VideoCapture first, fall back to a raw HTTP GET."""

    for attempt in range(1, MAX_ATTEMPTS + 1):

        print(f"[attempt {attempt}/{MAX_ATTEMPTS}] trying cv2.VideoCapture...")
        cap = cv2.VideoCapture(CAMERA_URL)
        if cap.isOpened():
            success, frame = cap.read()
            cap.release()
            if success and frame is not None:
                print("  -> got a frame via cv2.VideoCapture")
                return frame
        else:
            cap.release()

        print(f"[attempt {attempt}/{MAX_ATTEMPTS}] trying raw HTTP GET...")
        try:
            req = urllib.request.Request(
                CAMERA_URL, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                print("  -> got a frame via raw HTTP GET")
                return frame
        except Exception as exc:
            print(f"  -> failed: {exc}")

        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"Could not grab a frame from {CAMERA_URL} after "
        f"{MAX_ATTEMPTS} attempts."
    )


def build_lane_ranges(width):
    """Evenly split each direction's group into that many lanes."""

    ranges = {}

    down_start = int(width * LEFT_MARGIN_FRAC)
    down_end = int(width * DOWN_UP_GAP_START_FRAC)
    down_step = (down_end - down_start) / DOWN_LANES
    for i in range(DOWN_LANES):
        x_min = int(down_start + i * down_step)
        x_max = int(down_start + (i + 1) * down_step)
        ranges[f"lane_down_{i + 1}"] = (x_min, x_max)

    up_start = int(width * DOWN_UP_GAP_END_FRAC)
    up_end = int(width * RIGHT_MARGIN_FRAC)
    up_step = (up_end - up_start) / UP_LANES
    for i in range(UP_LANES):
        x_min = int(up_start + i * up_step)
        x_max = int(up_start + (i + 1) * up_step)
        ranges[f"lane_up_{i + 1}"] = (x_min, x_max)

    return ranges


def main():

    frame = grab_frame()
    height, width = frame.shape[:2]
    print(f"\nFrame size: {width} x {height}")

    line_y = round(height * LINE_Y_FRAC)
    print(
        f"LINE_Y = {line_y}  "
        f"(computed as {LINE_Y_FRAC:.1%} of this frame's height, "
        f"matching where LINE_Y=450 sat in the {REFERENCE_HEIGHT}px-tall "
        f"test video)"
    )
    if line_y >= height:
        print(
            f"WARNING: computed LINE_Y ({line_y}) is still >= frame "
            f"height ({height}) -- it won't be visible. Something is "
            f"off with REFERENCE_HEIGHT/REFERENCE_LINE_Y or the frame "
            f"grab returned an unexpectedly small image."
        )

    lane_ranges = build_lane_ranges(width)

    # Counting line
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
    cv2.putText(
        frame, f"LINE_Y={line_y}", (10, max(line_y - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )

    # Median strip between the two direction groups
    median_x1 = int(width * DOWN_UP_GAP_START_FRAC)
    median_x2 = int(width * DOWN_UP_GAP_END_FRAC)
    cv2.rectangle(frame, (median_x1, 0), (median_x2, height), (0, 165, 255), 1)
    cv2.putText(
        frame, "median", (median_x1 + 4, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1
    )

    # Lane boundaries, colored by direction: cyan = down, magenta = up
    print("\nLane boundaries (x-coordinates):")

    for lane_name, (x_min, x_max) in lane_ranges.items():

        print(f"  {lane_name}: {x_min} - {x_max}")

        is_down = lane_name.startswith("lane_down")
        color = (255, 255, 0) if is_down else (255, 0, 255)

        for x in (x_min, x_max):
            cv2.line(frame, (x, 0), (x, height), color, 1)

        label_x = (x_min + x_max) // 2
        cv2.putText(
            frame, lane_name, (label_x - 30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )

    # Direction arrows for a quick sanity check against the image
    cv2.putText(
        frame, "<- DOWN (away from camera)",
        (int(width * LEFT_MARGIN_FRAC), height - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
    )
    cv2.putText(
        frame, "UP (toward camera) ->",
        (int(width * DOWN_UP_GAP_END_FRAC), height - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, frame)

    print(f"\nSaved calibration image to: {OUTPUT_PATH}")
    print(
        "Open it and compare the cyan (down) / magenta (up) lines "
        "against the real lane markings. If they're off, adjust "
        "LEFT_MARGIN_FRAC / DOWN_UP_GAP_START_FRAC / "
        "DOWN_UP_GAP_END_FRAC / RIGHT_MARGIN_FRAC at the top of this "
        "script and rerun -- they're fractions of frame width, so "
        "they'll still make sense even if the feed's actual "
        "resolution isn't what you expect."
    )

    print("\nConfig values to paste into config.py once it looks right:")
    print(f"LINE_Y = {line_y}")
    print(f"LANE_COUNT = {DOWN_LANES + UP_LANES}")
    print("LANE_RANGES = {")
    for lane_name, (x_min, x_max) in lane_ranges.items():
        print(f'    "{lane_name}": ({x_min}, {x_max}),')
    print("}")


if __name__ == "__main__":
    main()
