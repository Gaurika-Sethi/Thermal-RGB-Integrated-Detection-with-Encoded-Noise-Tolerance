import cv2
import pandas as pd
from pathlib import Path


# Paths
video_path = "../../data/raw/rgb_vid.mp4"
csv_path = "../../outputs/rgb_multiple_people.csv"
output_path = "../../outputs/rgb_predictions.mp4"


# Load CSV
df = pd.read_csv(csv_path)

print("CSV loaded")
print("Total detections:", len(df))
print("Total frames:", df["frame"].nunique())


# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Resolution:", width, "x", height)


# Output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
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

    # Get detections for this frame
    frame_detections = df[df["frame"] == frame_number]

    person_count = 0

    for _, detection in frame_detections.iterrows():

        # Skip "no person" rows
        if pd.isna(detection["x"]):
            continue

        x_center = detection["x"]
        y_center = detection["y"]
        box_width = detection["width"]
        box_height = detection["height"]
        confidence = detection["confidence"]

        # Convert center coordinates to corners
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"Person {person_count + 1}: {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        person_count += 1

    # Frame information
    cv2.putText(
        frame,
        f"Frame: {frame_number} | Persons: {person_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    out.write(frame)

    frame_number += 1


cap.release()
out.release()

print()
print("Done!")
print("Annotated video saved to:")
print(output_path)