import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


def detect_people(video_path, output_csv):
    """
    Detect people in a thermal video using the trained YOLOv8 thermal model.

    Output CSV format:
    frame,x,y,width,height,confidence
    """

    project_root = Path(__file__).resolve().parents[2]

    model_path = project_root / "models" / "thermal" / "best.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Thermal model not found:\n{model_path}"
        )

    print("=" * 60)
    print("THERMAL PERSON DETECTOR")
    print("=" * 60)

    print(f"Loading model:\n{model_path}")

    model = YOLO(str(model_path))

    print("Model loaded successfully.\n")

    print(f"Opening video:\n{video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Frames : {total_frames}")
    print(f"FPS    : {fps:.2f}")
    print()

    detections = []

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % 10 == 0:
            print(f"Processing frame {frame_number}/{total_frames}")

        results = model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )

        boxes = results[0].boxes

        if len(boxes) == 0:

            detections.append([
                frame_number,
                None,
                None,
                None,
                None,
                0.0
            ])

            frame_number += 1
            continue

        best_box = max(
            boxes,
            key=lambda b: float(b.conf[0])
        )

        x1, y1, x2, y2 = best_box.xyxy[0].tolist()

        confidence = float(best_box.conf[0])

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        width = x2 - x1
        height = y2 - y1

        detections.append([
            frame_number,
            x_center,
            y_center,
            width,
            height,
            confidence
        ])

        frame_number += 1

    cap.release()

    df = pd.DataFrame(
        detections,
        columns=[
            "frame",
            "x",
            "y",
            "width",
            "height",
            "confidence"
        ]
    )

    output_csv = Path(output_csv)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output_csv, index=False)

    print()
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Frames processed : {len(df)}")
    print(f"CSV saved to      : {output_csv}")