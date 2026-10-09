
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
        (fused_x, fused_y, fused_confidence)
    """

    # Both modalities detected the person.
    if rgb_det is not None and thermal_det is not None:
        rgb_x, rgb_y, rgb_conf = rgb_det
        thermal_x, thermal_y, thermal_conf = thermal_det

        confidence_sum = rgb_conf + thermal_conf

        if confidence_sum == 0:
            return None

        fused_x = (
            rgb_conf * rgb_x + thermal_conf * thermal_x
        ) / confidence_sum

        fused_y = (
            rgb_conf * rgb_y + thermal_conf * thermal_y
        ) / confidence_sum

        fused_conf = confidence_sum / 2

        return fused_x, fused_y, fused_conf

    if rgb_det is not None:
        return rgb_det

    if thermal_det is not None:
        return thermal_det
    return None


if __name__ == "__main__":

    print("=" * 55)
    print("DAY 17 - SINGLE-MODALITY FUSION TESTS")
    print("=" * 55)

    # Test 1: RGB detects, thermal does not.
    rgb_det = (100.0, 200.0, 0.90)
    thermal_det = None

    result = fuse(rgb_det, thermal_det)

    print("\nTest 1: RGB only")
    print("Expected:", rgb_det)
    print("Actual:  ", result)

    assert result == rgb_det

    # Test 2: Thermal detects, RGB does not.
    rgb_det = None
    thermal_det = (250.0, 350.0, 0.80)

    result = fuse(rgb_det, thermal_det)

    print("\nTest 2: Thermal only")
    print("Expected:", thermal_det)
    print("Actual:  ", result)

    assert result == thermal_det

    print("\nBoth Day 17 tests passed.")
