# Dataset Notes — RGB Person Subset

## Dataset Purpose

This dataset is used for:

- RGB person detection training
- YOLOv8 fine-tuning
- reliability-system development later
- detector comparison experiments

The dataset is intentionally lightweight to support:

- Colab GPU workflow
- limited Google Drive storage
- CPU-first local development

---

# Dataset Structure

datasets/coco_subset/

```plaintext
images/
├── train/
└── val/

labels/
├── train/
└── val/

annotations/
├── instances_train2017.json
└── instances_val2017.json