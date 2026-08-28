def update_speed(
    track_id,
    center_y,
    frame_id,
    video_fps,
    speed_history,
    line1_y,
    line2_y,
    distance_m
):
    """
    Estimate a vehicle's speed from the time it takes to travel
    between two fixed reference lines (line1_y, line2_y).

    The previous implementation only recognized a line1 crossing
    when it matched the "down" pattern, and only started checking
    for line2 once state["direction"] had already been set. Two
    situations then silently produced no speed:

      1. A vehicle first detected already between (or past) the two
         lines never satisfies the "previous_y < line1_y <= center_y"
         style check for line1, so state["direction"] is never set,
         and line2's crossing is never checked at all -- even though
         we clearly saw it.
      2. A fast vehicle that jumps over both lines within a single
         frame (previous_y before line1, center_y already past
         line2) only registered the line1 crossing that frame; by
         the next frame center_y was already past line2_y too, so
         the line2 condition ("previous_y < line2_y") could never
         fire again.

    This version checks line1 and line2 crossings independently and
    directionally-agnostically every frame, so both cases above are
    captured correctly, and direction is inferred afterwards from
    whichever line was crossed first.
    """

    if video_fps <= 0:
        return None

    current_time = frame_id / video_fps

    # ---------------------------------------
    # Create vehicle state
    # ---------------------------------------

    if track_id not in speed_history:

        speed_history[track_id] = {
            "previous_y": center_y,
            "line1_time": None,
            "line2_time": None,
            "speed_kmh": None,
        }

        # Nothing to compare against yet.
        return None

    state = speed_history[track_id]

    previous_y = state["previous_y"]

    # ---------------------------------------
    # Line 1 crossing (either direction), first time only
    # ---------------------------------------

    if state["line1_time"] is None:

        crossed_down = previous_y < line1_y <= center_y
        crossed_up = previous_y > line1_y >= center_y

        if crossed_down or crossed_up:
            state["line1_time"] = current_time

    # ---------------------------------------
    # Line 2 crossing (either direction), first time only
    # ---------------------------------------

    if state["line2_time"] is None:

        crossed_down = previous_y < line2_y <= center_y
        crossed_up = previous_y > line2_y >= center_y

        if crossed_down or crossed_up:
            state["line2_time"] = current_time

    # ---------------------------------------
    # Calculate speed once both lines have been crossed
    # ---------------------------------------

    if (
        state["line1_time"] is not None
        and state["line2_time"] is not None
        and state["speed_kmh"] is None
    ):

        elapsed_time = abs(
            state["line2_time"] - state["line1_time"]
        )

        if elapsed_time > 0:

            speed_mps = distance_m / elapsed_time

            state["speed_kmh"] = speed_mps * 3.6

    # ---------------------------------------
    # Save current position
    # ---------------------------------------

    state["previous_y"] = center_y

    return state["speed_kmh"]
