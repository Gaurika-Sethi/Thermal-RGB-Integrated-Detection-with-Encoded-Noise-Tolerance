import torch
import pandas as pd


class DetectionFeatureBuilder:

    def __init__(self, modality_flag=1):
        """
        modality_flag:
            0 = RGB
            1 = Thermal
        """

        self.modality_flag = modality_flag

    def build_features(self, results, entropy_value):

        rows = []

        boxes = results[0].boxes

        if boxes is None:
            return rows

        for box in boxes:

            # Confidence
            confidence = float(box.conf.item())

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]

            width = float(x2 - x1)
            height = float(y2 - y1)

            # Area
            area = width * height

            # Aspect ratio
            aspect_ratio = width / (height + 1e-6)

            row = {
                "confidence": confidence,
                "bbox_area": area,
                "aspect_ratio": aspect_ratio,
                "entropy": entropy_value,
                "modality_flag": self.modality_flag
            }

            rows.append(row)

        return rows