# RGB YOLOv8s Training Notes

## Experiment Name

rgb_yolov8s_v1

---

# Dataset

COCO person-only subset

Train images: 5000
Val images: 1001

---

# Model

YOLOv8s pretrained

---

# Training Settings

Epochs: 20
Image size: 640
Batch size: 8

---

# Goals

- validate training pipeline
- observe detector learning
- establish baseline metrics

---

# Training Observations

## General

- training completed successfully
- GPU training worked
- dataset pipeline valid

---

# Metrics

Precision: 0.765
Recall: 0.610
mAP50: 0.709
mAP50-95: 0.465

---

# Observed Behavior

Examples:

- crowded scenes harder
- small humans occasionally missed
- confidence stable on clean images

---

# Infrastructure Notes

- Drive saving worked
- checkpoints saved correctly
- Colab runtime stable