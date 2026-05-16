# Week 3 Objective

Build and validate the first working thermal person detector using:

- LLVIP infrared dataset
- YOLOv8s
- Colab T4 GPU workflow

Focus:
- stable thermal detection pipeline
- dataset integrity
- inference validation

Not included yet:
- reliability estimation
- RT-DETR teacher
- entropy hooks
- Kalman integration

# Dataset Structure

Final YOLO dataset:

LLVIP_YOLO/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── thermal.yaml

Dataset statistics:
- Train images: 12025
- Train labels: 12025
- Valid images: 3463
- Valid labels: 3463

Only infrared modality used for Week 3.

# Training Configuration

Model:
- YOLOv8s

Environment:
- Google Colab T4 GPU

Parameters:
- epochs: 30
- imgsz: 640
- batch: 16
- workers: 2

Checkpoint strategy:
- best.pt used as final stable detector
- last.pt became unstable during late training

# Pipeline Validation

A small dev_slice (~50 images) was used to verify:

- dataset loading
- annotation correctness
- GPU training
- validation execution
- inference execution

Result:
Pipeline operated successfully without NaNs or dataset corruption.

# Detector Behavior Observations

Thermal detector strengths:
- strong nighttime detection
- clear human silhouette detection
- good performance in high thermal contrast scenes

Observed weaknesses:
- unstable confidence in crowded scenes
- occasional false positives
- difficulty with tiny/distant people
- partial occlusion failures

# Failure Cases

Observed issue:
- late-stage training instability after extending epochs

Symptoms:
- excessive detections
- NMS time limit exceeded
- RAM crash during inference
- unstable predictions from last.pt

Resolution:
- reverted to best.pt
- limited max_det during inference
- increased confidence threshold

# Motivation for Reliability Stage

Week 3 observations showed that confidence alone is insufficient for stable perception.

Examples:
- unstable high-confidence detections
- excessive detections during overtraining
- NMS overload caused by noisy predictions

These observations justify the need for:
- reliability estimation
- confidence refinement
- reliability-aware tracking

# Final Outputs

Generated artifacts:
- thermal_best.pt
- inference_outputs/
- results.png
- thermal.yaml
- LLVIP_YOLO.zip

Drive structure finalized for future runtime recovery.