import cv2
import pandas as pd


def create_overlay(video_path, csv_path, output_path):

    detections = pd.read_csv(csv_path)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        row = detections.iloc[frame_number]

        if row["confidence"] > 0:

            x = int(row["x"])
            y = int(row["y"])

            w = int(row["width"])
            h = int(row["height"])

            x1 = x - w // 2
            y1 = y - h // 2

            x2 = x + w // 2
            y2 = y + h // 2

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"{row['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        frame_number += 1

    cap.release()
    writer.release()

    print("Overlay video saved.")