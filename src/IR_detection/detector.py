
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# HARDCODED PATHS
# ============================================================

MODEL_PATH = Path(
    r"C:\thermal project\Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance\models\thermal\best.pt"
)

VIDEO_PATH = Path(
    r"C:\thermal project\Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance\data\raw\IR_light_vid.mp4"
)

OUTPUT_CSV = Path(
    r"C:\thermal project\Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance\outputs\IR_light_detection.csv"
)

CONFIDENCE_THRESHOLD = 0.25


def detect_people():
    """
    Detect all people in a thermal/IR video.

    CSV columns:
    frame, x, y, width, height, confidence

    Multiple people per frame are supported.
    Frames without detections receive one row with confidence 0.
    """

    print("=" * 60)
    print("THERMAL / IR MULTI-PERSON DETECTOR")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(f"\nLoading model:\n{MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    print(f"\nOpening video:\n{VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Total frames : {total_frames}")
    print(f"FPS          : {fps:.2f}")
    print(f"Resolution   : {frame_width} x {frame_height}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print()

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    detections = []
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_number % 50 == 0:
                print(
                    f"Processing frame "
                    f"{frame_number}/{total_frames}"
                )

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            person_count = 0

            for box in results[0].boxes:
                # Class 0 represents person in this model.
                if int(box.cls[0]) != 0:
                    continue

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                box_width = x2 - x1
                box_height = y2 - y1

                detections.append([
                    frame_number,
                    x_center,
                    y_center,
                    box_width,
                    box_height,
                    confidence
                ])

                person_count += 1

            # Keep frames with no detections in the CSV.
            if person_count == 0:
                detections.append([
                    frame_number,
                    None,
                    None,
                    None,
                    None,
                    0.0
                ])

            frame_number += 1

    finally:
        cap.release()

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    columns = [
        "frame",
        "x",
        "y",
        "width",
        "height",
        "confidence"
    ]

    df = pd.DataFrame(
        detections,
        columns=columns
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Report results
    # --------------------------------------------------------

    detection_rows = int(
        (df["confidence"] > 0).sum()
    )

    detected_frames = int(
        df.loc[
            df["confidence"] > 0,
            "frame"
        ].nunique()
    )

    no_detection_rows = int(
        (df["confidence"] == 0).sum()
    )

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)

    print(f"Frames processed       : {frame_number}")
    print(f"Detection rows         : {detection_rows}")
    print(f"Frames with detections : {detected_frames}")
    print(f"No-detection rows      : {no_detection_rows}")
    print(f"Total CSV rows         : {len(df)}")
    print(f"\nCSV saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    detect_people()
