import cv2
import trajectory as traj


def display_output(
    result,
    current_vehicle_count,
    vehicle_counts,
    track_history,
    LINE_Y,
    vehicles_passed,
    processing_fps,
    direction_counts,
    total_vehicle_counts,
    vehicles_per_hour,
    vehicle_speeds,
    lane_counts,
    lane_ranges
):

    annotated_frame = result.plot()

    # -----------------------------------
    # FPS
    # -----------------------------------

    cv2.putText(
        annotated_frame,
        f"Inference FPS: {processing_fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # -----------------------------------
    # Current vehicles
    # -----------------------------------

    y = 60

    cv2.putText(
        annotated_frame,
        f"Vehicles: {current_vehicle_count}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    y += 30

    # -----------------------------------
    # Vehicle types
    # -----------------------------------

    for vehicle_type, count in vehicle_counts.items():

        cv2.putText(
            annotated_frame,
            f"{vehicle_type}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y += 30

    # -----------------------------------
    # Trajectories
    # -----------------------------------

    traj.draw_trajectory(
        track_history,
        annotated_frame
    )

    # -----------------------------------
    # Lane boundaries (for calibration)
    # -----------------------------------

    draw_lane_boundaries(
        annotated_frame,
        lane_ranges,
        LINE_Y
    )

    # -----------------------------------
    # Counting line
    # -----------------------------------

    cv2.line(
        annotated_frame,
        (0, LINE_Y),
        (
            annotated_frame.shape[1],
            LINE_Y
        ),
        (0, 0, 255),
        2
    )

    # -----------------------------------
    # Vehicles passed
    # -----------------------------------

    y += 20

    cv2.putText(
        annotated_frame,
        f"Passed: {vehicles_passed}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    y += 30

    # -----------------------------------
    # Vehicle types
    # -----------------------------------

    for vehicle_type, count in total_vehicle_counts.items():

        cv2.putText(
            annotated_frame,
            f"{vehicle_type}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y += 30

    y += 30

    cv2.putText(
        annotated_frame,
        f"Direction:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    y += 30

    cv2.putText(
        annotated_frame,
        f"Up: {direction_counts['up']}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    y += 30

    cv2.putText(
        annotated_frame,
        f"Down: {direction_counts['down']}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    y += 50

    # -----------------------------------
    # Vehicle speed
    # -----------------------------------

    if result.boxes.id is not None:

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        boxes_by_id = dict(zip(track_ids, result.boxes))

        for track_id, speed_kmh in vehicle_speeds.items():

            box = boxes_by_id.get(track_id)

            if box is None:
                continue

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            text_x = int(x1)
            text_y = int(y1) - 10

            cv2.putText(
                annotated_frame,
                f"{speed_kmh:.1f} km/h",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

    # -----------------------------------
    # Lane traffic flow
    # -----------------------------------

    y += 30

    cv2.putText(
        annotated_frame,
        "LANE FLOW:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    y += 30

    for lane, counts in lane_counts.items():

        text = (
            f"{lane}: "
            f"Total {counts['total']} "
            f"Up {counts['up']} "
            f"Down {counts['down']}"
        )

        color = (0, 255, 0) if lane != "unassigned" else (0, 165, 255)

        cv2.putText(
            annotated_frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        y += 25

    y += 30

    cv2.putText(
        annotated_frame,
        f"Flow: {vehicles_per_hour:.0f} vehicles/hour",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return annotated_frame


def draw_lane_boundaries(
    annotated_frame,
    lane_ranges,
    line_y
):
    """
    Draw each lane's x-range as vertical guide lines plus a label
    near LINE_Y. This exists so you can visually check the pixel
    values in config.LANE_RANGES actually line up with the real
    lane markings in the video -- run the video, pause near LINE_Y,
    and nudge the ranges in config.py until the drawn boundaries
    sit on the lane dividers.
    """

    height = annotated_frame.shape[0]

    drawn_x = set()

    for lane_name, (x_min, x_max) in lane_ranges.items():

        for x in (x_min, x_max):

            if x in drawn_x:
                continue

            drawn_x.add(x)

            cv2.line(
                annotated_frame,
                (x, 0),
                (x, height),
                (255, 0, 255),
                1
            )

        label_x = (x_min + x_max) // 2

        cv2.putText(
            annotated_frame,
            lane_name,
            (label_x - 20, max(line_y - 15, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1
        )
