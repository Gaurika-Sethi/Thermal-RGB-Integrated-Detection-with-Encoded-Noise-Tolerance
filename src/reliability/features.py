def extract_detection_features(detection):

    x1, y1, x2, y2 = detection["bbox"]

    width = x2 - x1
    height = y2 - y1

    area = width * height

    # avoid divide-by-zero
    aspect_ratio = width / (height + 1e-6)

    modality_map = {
        "rgb": 0,
        "thermal": 1
    }

    modality_flag = modality_map.get(
        detection["modality"].lower(),
        -1
    )

    features = {
        "bbox": detection["bbox"],
        "confidence": detection["confidence"],
        "area": round(area, 2),
        "aspect_ratio": round(aspect_ratio, 4),
        "modality": modality_flag
    }

    return features