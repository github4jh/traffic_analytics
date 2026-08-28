import config


def analyze_frame(
    result,
    vehicle_counts,
    track_history,
    counted_ids,
    vehicles_passed,
    LINE_Y,
    direction_counts,
    total_vehicle_counts,
    lane_counts,
    track_side,
):

    # ---------------------------------------
    # No tracking IDs
    # ---------------------------------------

    if result.boxes.id is None:
        return vehicles_passed

    # ---------------------------------------
    # Get tracking IDs
    # ---------------------------------------

    track_ids = (
        result.boxes.id
        .int()
        .cpu()
        .tolist()
    )

    # ---------------------------------------
    # Process every tracked object
    # ---------------------------------------

    for track_id, box in zip(
        track_ids,
        result.boxes
    ):

        # -----------------------------------
        # Vehicle class
        # -----------------------------------

        class_id = int(box.cls[0])

        if class_id not in config.VEHICLE_CLASSES:
            continue

        class_name = config.VEHICLE_CLASSES[
            class_id
        ]

        vehicle_counts[class_name] += 1

        # -----------------------------------
        # Bounding box center
        # -----------------------------------

        x1, y1, x2, y2 = (
            box.xyxy[0].tolist()
        )

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(y2)

        # -----------------------------------
        # Store trajectory
        # -----------------------------------

        track_history[track_id].append(
            (center_x, center_y)
        )

        # -----------------------------------
        # Detect line crossing
        # -----------------------------------

        direction = detect_crossing(
            track_side,
            track_id,
            center_y,
            LINE_Y,
            config.LINE_MARGIN
        )

        # -----------------------------------
        # Count vehicle only once
        # -----------------------------------

        if direction is not None:

            if track_id not in counted_ids:

                # Remember that this vehicle
                # has already been counted.
                counted_ids.add(track_id)

                # Total vehicles passed
                vehicles_passed += 1

                # Direction
                direction_counts[direction] += 1

                total_vehicle_counts[class_name] += 1

                lane = get_lane(
                    center_x,
                    config.LANE_RANGES
                )

                # Vehicles that cross the line outside every
                # configured lane range (e.g. gaps between ranges,
                # or the median) still get counted overall, but are
                # bucketed separately instead of silently dropped.
                # A growing "unassigned" count is a strong signal
                # that LANE_RANGES needs recalibrating.
                lane_key = lane if lane is not None else "unassigned"

                lane_counts[lane_key]["total"] += 1
                lane_counts[lane_key][direction] += 1

    return vehicles_passed


def detect_crossing(
    track_side,
    track_id,
    center_y,
    LINE_Y,
    LINE_MARGIN=10
):
    """
    Determine whether a tracked vehicle has crossed LINE_Y.

    This uses a persistent per-track "side" state (above/below
    LINE_Y) with hysteresis, instead of comparing only the last two
    trajectory points. The old two-point approach required a vehicle
    to jump from clearly-above to clearly-below LINE_Y within a
    single pair of consecutive frames; any vehicle whose per-frame
    movement was smaller than the margin band would pass through
    that dead zone across several frames and never trigger either
    condition, which is why the count came out far lower than the
    real traffic volume.

    Here, a crossing fires as soon as the vehicle is clearly on the
    opposite side from where we last recorded it, no matter how many
    frames it took to get there.

    Returns:
        "up"   -> vehicle crossed upward
        "down" -> vehicle crossed downward
        None   -> no crossing
    """

    upper_limit = LINE_Y - LINE_MARGIN
    lower_limit = LINE_Y + LINE_MARGIN

    current_side = track_side.get(track_id)

    # First time we see this track: just record which side of the
    # line it started on, no crossing yet.
    if current_side is None:
        track_side[track_id] = (
            "above" if center_y < LINE_Y else "below"
        )
        return None

    direction = None

    if current_side == "above" and center_y > lower_limit:
        direction = "down"
        track_side[track_id] = "below"

    elif current_side == "below" and center_y < upper_limit:
        direction = "up"
        track_side[track_id] = "above"

    return direction


def get_lane(
    center_x,
    lane_ranges
):

    for lane_name, (
        x_min,
        x_max
    ) in lane_ranges.items():

        if (
            x_min <= center_x < x_max
        ):
            return lane_name

    return None
