import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.insert(0, PROJECT_ROOT)

from src.matching import match_detections


# =========================================================
# DUMMY DATA
# =========================================================

yolo_boxes = [
    [100, 100, 200, 300],
    [400, 400, 500, 600]
]

teacher_boxes = [
    [105, 105, 205, 305],
    [700, 700, 800, 800]
]

teacher_scores = [0.91, 0.72]

# =========================================================
# MATCH
# =========================================================

matches = match_detections(
    yolo_boxes,
    teacher_boxes,
    teacher_scores
)

# =========================================================
# PRINT
# =========================================================

for m in matches:
    print(m)