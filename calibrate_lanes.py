"""
Lane calibration helper.

The per-lane counts (bug 2) can only ever be as correct as
config.LANE_RANGES, and those are just pixel x-coordinates someone
guessed -- there's no way to derive them from the video without
looking at it. This script grabs a single frame from config.LOCAL,
overlays LINE_Y, the two speed lines, and every LANE_RANGES boundary
with labels, and saves it so you can compare against the real lane
markings and adjust config.py until they line up.

Usage:
    python calibrate_lanes.py
    -> writes output/lane_calibration.png
    -> prints the boundary x-coordinates for reference
"""

import os
import cv2

import config


def main():

    cap = cv2.VideoCapture(config.LOCAL)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {config.LOCAL}")

    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        raise RuntimeError(f"Could not read a frame from: {config.LOCAL}")

    height, width = frame.shape[:2]

    # Counting line
    cv2.line(frame, (0, config.LINE_Y), (width, config.LINE_Y), (0, 0, 255), 2)
    cv2.putText(
        frame, f"LINE_Y={config.LINE_Y}", (10, config.LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )

    # Speed lines
    for label, y in (
        ("SPEED_LINE_1_Y", config.SPEED_LINE_1_Y),
        ("SPEED_LINE_2_Y", config.SPEED_LINE_2_Y),
    ):
        cv2.line(frame, (0, y), (width, y), (255, 255, 0), 1)
        cv2.putText(
            frame, f"{label}={y}", (10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
        )

    # Lane boundaries
    print("Lane boundaries (x-coordinates):")

    drawn_x = set()

    for lane_name, (x_min, x_max) in config.LANE_RANGES.items():

        print(f"  {lane_name}: {x_min} - {x_max}")

        for x in (x_min, x_max):
            if x not in drawn_x:
                drawn_x.add(x)
                cv2.line(frame, (x, 0), (x, height), (255, 0, 255), 1)

        label_x = (x_min + x_max) // 2
        cv2.putText(
            frame, lane_name, (label_x - 25, config.LINE_Y - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
        )

    os.makedirs(os.path.dirname(config.OUTPUT), exist_ok=True)

    out_path = os.path.join(
        os.path.dirname(config.OUTPUT), "lane_calibration.png"
    )

    cv2.imwrite(out_path, frame)

    print(f"\nFrame size: {width} x {height}")
    print(f"Saved calibration image to: {out_path}")
    print(
        "Open it, compare the magenta lines against the real lane "
        "markings, and edit LANE_RANGES in config.py until they match."
    )


if __name__ == "__main__":
    main()
