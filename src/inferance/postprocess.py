from src.reliability.features import extract_detection_features


def postprocess_detections(
    detections,
    confidence_threshold=0.4
):

    processed = []

    for detection in detections:

        confidence = detection["confidence"]

        # =====================
        # CONFIDENCE FILTER
        # =====================

        if confidence < confidence_threshold:
            continue

        # =====================
        # FEATURE PACKAGING
        # =====================

        features = extract_detection_features(
            detection
        )

        processed.append(features)

    return processed