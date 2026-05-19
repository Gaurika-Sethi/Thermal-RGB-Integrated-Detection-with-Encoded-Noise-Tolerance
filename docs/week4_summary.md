# Week 4 Day 3 — Batch Inference Notes

## Objective

Validate stable batch inference on LLVIP thermal validation images using the modular inference pipeline.

---

# Runtime Configuration

Model:
- YOLOv8s thermal detector

Checkpoint:
- thermal_best.pt

Inference Settings:
- confidence_threshold = 0.4
- max_det = 20
- imgsz = 640

Dataset:
- LLVIP thermal validation set

Initial Batch Size:
- 20 images

---

# Pipeline Components Tested

The following modules were tested together:

- detector wrapper
- feature extraction
- visualization pipeline
- metadata export
- batch inference loop

---

# Successful Behaviors

Observed strengths:

- inference completed without crashes
- no NMS overflow observed
- structured detection outputs generated correctly
- metadata JSON exported successfully
- visualization images saved correctly

Thermal detector strengths observed:

- strong large-person detection
- stable nighttime silhouettes
- good high-contrast thermal detection

---

# Failure Cases / Weaknesses

Observed issues:

- occasional false positives
- unstable confidence on distant humans
- some crowded scenes produced overlapping boxes
- tiny people occasionally missed

Any strange runtime behavior:

- [write if any]
- [memory spikes?]
- [slow inference?]
- [duplicate detections?]

---

# Empty Prediction Analysis

Number of images with zero detections:
- [your value]

Observations:
- empty outputs handled correctly
- pipeline did not crash on no-detection images

---

# Metadata Validation

Extracted features:

- bbox
- confidence
- area
- aspect_ratio
- modality flag

Metadata format validated successfully.

---

# Key Insight

The detector pipeline is now capable of:

image
→ structured detections
→ feature extraction
→ metadata export
→ visualization generation

This prepares the system for:
- reliability estimation
- entropy integration
- RT-DETR teacher supervision
- future Kalman tracking

---

# Next Steps

Planned next stage:
- entropy hook infrastructure
- intermediate feature extraction
- reliability-ready feature expansion

# Week 4 Day 4 — Entropy Hook Infrastructure

## Objective

Validate that intermediate YOLO feature tensors can be extracted using forward hooks.

---

# Tested Components

- PyTorch forward hooks
- YOLO intermediate activations
- tensor extraction pipeline

---

# Results

- hooks attached successfully
- feature tensors captured correctly
- no inference crashes observed
- tensor shapes varied across network depth

---

# Observations

Early layers:
- large spatial resolution
- fewer channels

Deeper layers:
- smaller spatial resolution
- more semantic channels

---

# Important Outcome

The system can now access intermediate feature activations required for future entropy estimation.

No entropy calculations were performed yet.

# Week 4 Day 5 — Runtime Modularization + Router Logic

## Objective

Prepare a modular multi-modality inference pipeline capable of routing RGB and thermal inputs to the correct detector.

---

# Components Implemented

## 1. Detection Router

Created:

- src/inference/router.py

Purpose:
- route input frames based on modality metadata
- avoid learned routing complexity
- maintain deterministic behavior

Routing logic used:

- RGB → RGB detector
- thermal → thermal detector

No classifier-based routing was used.

---

## 2. Postprocessing Module

Created:

- src/inference/postprocess.py

Responsibilities:

- confidence filtering
- structured bbox formatting
- feature packaging
- compatibility with reliability pipeline

---

# Runtime Pipeline

Current runtime flow:

Input Image
→ Router
→ Detector
→ Postprocessing
→ Feature Extraction
→ Structured Output

---

# RGB Pipeline Validation

RGB validation images from COCO subset were tested.

Observed behavior:

- RGB detector loaded successfully
- router selected correct detector
- outputs generated with modality = 0
- structured outputs remained consistent

Example output structure:

{
    "bbox": [...],
    "confidence": ...,
    "area": ...,
    "aspect_ratio": ...,
    "modality": 0
}

---

# Thermal Pipeline Validation

Thermal validation images from LLVIP were tested.

Observed behavior:

- thermal detector loaded correctly
- outputs generated with modality = 1
- feature extraction remained stable

---

# Architectural Improvements

The project is now modularized into separate responsibilities:

- detector.py → inference only
- router.py → modality selection
- postprocess.py → runtime cleanup
- features.py → feature extraction

This reduces notebook dependency and improves maintainability.

---

# Important Outcome

The system now supports modality-aware runtime inference using reusable modules instead of isolated notebook logic.

This prepares the architecture for:

- entropy integration
- reliability estimation
- fusion logic
- Kalman tracking
- future deployment pipeline

---

# Current System Status

The runtime pipeline can now perform:

image
→ modality routing
→ detector inference
→ feature extraction
→ structured metadata generation

for both RGB and thermal modalities.

# Automated Metadata Analysis

Detector outputs were analyzed automatically using structured metadata statistics.

Observed statistics:

- Total detections: 63
- Average confidence: 0.7754
- Low-confidence detections: 0
- Weird aspect ratio detections: 1
- Crowded images detected: 5

Key observations:

- detector outputs remained stable during batch inference
- no major detection explosion observed
- no severe localization corruption observed
- crowded scenes remain potentially difficult cases

One detection with unusual aspect ratio was identified:
- 190013.jpg → aspect_ratio = 1.5146

This suggests that metadata-based heuristics can help identify suspicious detections before full reliability modeling is introduced.