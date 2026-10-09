
def fuse(rgb_det, thermal_det):
    """
    Parameters
    ----------
    rgb_det : tuple or None
        (x, y, confidence)

    thermal_det : tuple or None
        (x, y, confidence)

    Returns
    -------
    tuple or None
        (fused_x, fused_y, fused_confidence),
        or None if no valid measurement is available.
    """

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

        return fused_x, fused_y, fused_conf

    # Only RGB detects the person.
    if rgb_det is not None:
        return rgb_det

    # Only thermal detects the person.
    if thermal_det is not None:
        return thermal_det

    return None


if __name__ == "__main__":

    print("=" * 55)
    print("DAY 18 - MISSING MEASUREMENT TEST")
    print("=" * 55)

    # Neither modality detects a person.
    rgb_det = None
    thermal_det = None

    result = fuse(rgb_det, thermal_det)

    print("\nTest: Neither modality detects")
    print("RGB:     ", rgb_det)
    print("Thermal: ", thermal_det)
    print("Expected:", None)
    print("Actual:  ", result)

    assert result is None

    print("\nDay 18 test passed.")
