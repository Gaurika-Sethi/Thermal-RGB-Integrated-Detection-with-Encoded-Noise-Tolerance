def fuse(rgb_det, thermal_det):
    """
    Fuse one RGB detection and one thermal detection.

    Parameters
    ----------
    rgb_det : tuple or None
        RGB detection in the form:
        (x, y, confidence)

    thermal_det : tuple or None
        Thermal detection in the form:
        (x, y, confidence)

    Returns
    -------
    tuple or None
        (fused_x, fused_y, fused_confidence)

        Returns None if neither modality has a detection.
    """

    # ---------------------------------------------------------
    # Case 1:
    # Both RGB and thermal detected the person.
    # ---------------------------------------------------------
    if rgb_det is not None and thermal_det is not None:

        rgb_x, rgb_y, rgb_conf = rgb_det
        thermal_x, thermal_y, thermal_conf = thermal_det

        confidence_sum = rgb_conf + thermal_conf
        if confidence_sum == 0:
            return None

        fused_x = (
            rgb_conf * rgb_x
            + thermal_conf * thermal_x
        ) / confidence_sum

        fused_y = (
            rgb_conf * rgb_y
            + thermal_conf * thermal_y
        ) / confidence_sum

        fused_conf = confidence_sum / 2

        return (
            fused_x,
            fused_y,
            fused_conf
        )

    # ---------------------------------------------------------
    # Case 2:
    # Only RGB detected the person.
    # Trust the RGB position directly.
    # ---------------------------------------------------------
    if rgb_det is not None:
        return rgb_det

    # ---------------------------------------------------------
    # Case 3:
    # Only thermal detected the person.
    # Trust the thermal position directly.
    # ---------------------------------------------------------
    if thermal_det is not None:
        return thermal_det

    # ---------------------------------------------------------
    # Case 4:
    # Neither modality detected the person.
    #
    # Do NOT create a position.
    # The future Kalman filter will handle this.
    # ---------------------------------------------------------
    return None


def create_detection(x, y, confidence):
    """
    Create a standard detection tuple.

    Returns
    -------
    tuple
        (x, y, confidence)
    """

    return (
        float(x),
        float(y),
        float(confidence)
    )


if __name__ == "__main__":

    print("=" * 60)
    print("CONFIDENCE-WEIGHTED FUSION TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Test 1: Both modalities detect
    # ---------------------------------------------------------

    rgb_det = create_detection(
        100,
        200,
        0.9
    )

    thermal_det = create_detection(
        200,
        300,
        0.3
    )

    result = fuse(
        rgb_det,
        thermal_det
    )

    print("\nTest 1 - Both detected")

    print(f"RGB     : {rgb_det}")
    print(f"Thermal : {thermal_det}")
    print(f"Fused   : {result}")

    # ---------------------------------------------------------
    # Test 2: Only RGB detects
    # ---------------------------------------------------------

    rgb_det = create_detection(
        100,
        200,
        0.9
    )

    thermal_det = None

    result = fuse(
        rgb_det,
        thermal_det
    )

    print("\nTest 2 - RGB only")

    print(f"RGB     : {rgb_det}")
    print(f"Thermal : {thermal_det}")
    print(f"Fused   : {result}")

    # ---------------------------------------------------------
    # Test 3: Only thermal detects
    # ---------------------------------------------------------

    rgb_det = None

    thermal_det = create_detection(
        200,
        300,
        0.8
    )

    result = fuse(
        rgb_det,
        thermal_det
    )

    print("\nTest 3 - Thermal only")

    print(f"RGB     : {rgb_det}")
    print(f"Thermal : {thermal_det}")
    print(f"Fused   : {result}")

    # ---------------------------------------------------------
    # Test 4: Neither detects
    # ---------------------------------------------------------

    rgb_det = None
    thermal_det = None

    result = fuse(
        rgb_det,
        thermal_det
    )

    print("\nTest 4 - Neither detected")

    print(f"RGB     : {rgb_det}")
    print(f"Thermal : {thermal_det}")
    print(f"Fused   : {result}")

    print()
    print("=" * 60)
    print("FUSION TESTS COMPLETE")
    print("=" * 60)