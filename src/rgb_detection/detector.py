import cv2
import pandas as pd
from ultralytics import YOLO

def detect_people(video_path, output_csv):

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    print("Video opened:", cap.isOpened())

    detections = []

    frame_number = 0

    while True:

        ret, frame = cap.read()

        print(f"Frame {frame_number}: ret={ret}")

        if not ret:
            break

        print("Running YOLO...")

        results = model(frame, verbose=False)

        print("Boxes:", len(results[0].boxes))

        person_found = False

        for box in results[0].boxes:

            cls = int(box.cls[0])

            print("Class:", cls)

            if cls != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            confidence = float(box.conf[0])

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

            print("Person detected!")

            person_found = True
            break

        if not person_found:

            print("No person")

            detections.append([
                frame_number,
                None,
                None,
                None,
                None,
                0
            ])

        frame_number += 1

    cap.release()

    print("Final length:", len(detections))

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

    print(df.head())

    df.to_csv(output_csv, index=False)

    print("Done")