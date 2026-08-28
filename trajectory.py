import cv2


def draw_trajectory(
    track_history,
    annotated_frame
):
    """
    Draw the historical trajectory of each tracked vehicle.

    Args:
        track_history:
            Dictionary mapping track IDs to a sequence of
            (x, y) center coordinates.

        annotated_frame:
            OpenCV image on which trajectories are drawn.
    """

    for track_id, history in track_history.items():

        # Need at least two points to draw a line
        if len(history) < 2:
            continue

        pts = list(history)

        # Draw trajectory segments
        for i in range(1, len(pts)):

            cv2.line(
                annotated_frame,
                pts[i - 1],
                pts[i],
                (0, 255, 255),
                2
            )

        # Draw the current center point
        current_x, current_y = pts[-1]

        cv2.circle(
            annotated_frame,
            (current_x, current_y),
            4,
            (0, 255, 255),
            -1
        )

        # Display track ID near the current position
        cv2.putText(
            annotated_frame,
            f"ID: {track_id}",
            (current_x + 5, current_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1
        )