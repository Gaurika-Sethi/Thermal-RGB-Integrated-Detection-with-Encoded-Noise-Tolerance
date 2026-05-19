from ultralytics import YOLO
import cv2


class ThermalDetector:

    def __init__(
        self,
        model_path,
        confidence_threshold=0.4,
        max_det=20,
        image_size=640,
        modality="thermal"
    ):

        self.model = YOLO(model_path)

        self.confidence_threshold = confidence_threshold
        self.max_det = max_det
        self.image_size = image_size
        self.modality = modality

    def predict(self, image_path):

        results = self.model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            max_det=self.max_det,
            imgsz=self.image_size,
            verbose=False
        )

        detections = []

        if len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:

            xyxy = box.xyxy[0].tolist()

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            detection = {
                "bbox": [
                    round(xyxy[0], 2),
                    round(xyxy[1], 2),
                    round(xyxy[2], 2),
                    round(xyxy[3], 2)
                ],
                "confidence": round(confidence, 4),
                "class_id": class_id,
                "modality": self.modality
            }

            detections.append(detection)

        return detections