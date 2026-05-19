import numpy as np


# =========================================================
# IOU
# =========================================================

def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)

    inter_area = inter_w * inter_h

    boxA_area = (
        (boxA[2] - boxA[0]) *
        (boxA[3] - boxA[1])
    )

    boxB_area = (
        (boxB[2] - boxB[0]) *
        (boxB[3] - boxB[1])
    )

    union = boxA_area + boxB_area - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


# =========================================================
# MATCH DETECTIONS
# =========================================================

def match_detections(
    yolo_boxes,
    teacher_boxes,
    teacher_scores,
    iou_thresh=0.5
):

    matches = []

    for i, yolo_box in enumerate(yolo_boxes):

        best_iou = 0
        best_score = 0

        for j, teacher_box in enumerate(teacher_boxes):

            iou = compute_iou(
                yolo_box,
                teacher_box
            )

            if iou > best_iou:

                best_iou = iou
                best_score = teacher_scores[j]

        if best_iou >= iou_thresh:

            matches.append({
                "yolo_idx": i,
                "matched": True,
                "iou": best_iou,
                "rtdetr_conf": best_score
            })

        else:

            matches.append({
                "yolo_idx": i,
                "matched": False,
                "iou": best_iou,
                "rtdetr_conf": 0.3
            })

    return matches