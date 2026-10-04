import cv2
import pandas as pd
from pathlib import Path


def visualize_detections(video_path, csv_path, output_video):
    """
    Draw detections from a CSV file onto the original video.

    CSV format:
        frame,x,y,width,height,confidence

    Multiple detections per frame are supported.
    """

    video_path = Path(video_path)
    csv_path = Path(csv_path)
    output_video = Path(output_video)

    # --------------------------------------------------
    # Validate input files
    # --------------------------------------------------

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{csv_path}"
        )

    # --------------------------------------------------
    # Load CSV
    # --------------------------------------------------

    print("=" * 60)
    print("THERMAL DETECTION VISUALIZER")
    print("=" * 60)

    print(f"\nLoading CSV:\n{csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "frame",
        "x",
        "y",
        "width",
        "height",
        "confidence"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"CSV is missing required column: {column}"
            )

    print(f"CSV rows: {len(df)}")

    # Group detections by frame.
    # This is important because multiple people
    # can have the same frame number.
    detections_by_frame = {
        int(frame): group
        for frame, group in df.groupby("frame")
    }

    # --------------------------------------------------
    # Open video
    # --------------------------------------------------

    print(f"\nOpening video:\n{video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Width  : {width}")
    print(f"Height : {height}")
    print(f"FPS    : {fps:.2f}")
    print(f"Frames : {total_frames}")

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Video writer
    # --------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video:\n{output_video}"
        )

    # --------------------------------------------------
    # Process frames
    # --------------------------------------------------

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % 50 == 0:
            print(
                f"Visualizing frame "
                f"{frame_number}/{total_frames}"
            )

        # Get all detections belonging to this frame.
        frame_detections = detections_by_frame.get(
            frame_number,
            []
        )

        detection_count = 0

        for _, detection in frame_detections.iterrows():

            confidence = float(
                detection["confidence"]
            )

            # Skip no-detection rows.
            if confidence <= 0:
                continue

            x_center = float(
                detection["x"]
            )

            y_center = float(
                detection["y"]
            )

            box_width = float(
                detection["width"]
            )

            box_height = float(
                detection["height"]
            )

            # --------------------------------------------------
            # Convert center/width/height to corner coordinates
            # --------------------------------------------------

            x1 = int(
                x_center - box_width / 2
            )

            y1 = int(
                y_center - box_height / 2
            )

            x2 = int(
                x_center + box_width / 2
            )

            y2 = int(
                y_center + box_height / 2
            )

            # Keep coordinates inside the image.
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            # --------------------------------------------------
            # Draw bounding box
            # --------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # --------------------------------------------------
            # Confidence label
            # --------------------------------------------------

            label = (
                f"Person {confidence:.2f}"
            )

            text_y = max(
                y1 - 10,
                20
            )

            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # --------------------------------------------------
            # Draw center point
            # --------------------------------------------------

            cv2.circle(
                frame,
                (
                    int(x_center),
                    int(y_center)
                ),
                4,
                (0, 0, 255),
                -1
            )

            detection_count += 1

        # --------------------------------------------------
        # Frame information
        # --------------------------------------------------

        info_text = (
            f"Frame: {frame_number} | "
            f"People: {detection_count}"
        )

        cv2.putText(
            frame,
            info_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # --------------------------------------------------
        # Write frame
        # --------------------------------------------------

        writer.write(frame)

        frame_number += 1

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    cap.release()
    writer.release()

    print()
    print("=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)

    print(f"Frames processed : {frame_number}")
    print(f"Output video     : {output_video}")


# ------------------------------------------------------
# Main
# ------------------------------------------------------

if __name__ == "__main__":

    PROJECT_ROOT = Path(
        __file__
    ).resolve().parents[2]

    VIDEO_PATH = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "thermal_vid.mp4"
    )

    CSV_PATH = (
        PROJECT_ROOT
        / "outputs"
        / "thermal_multiple_people.csv"
    )

    OUTPUT_VIDEO = (
        PROJECT_ROOT
        / "outputs"
        / "thermal_multi_person_verified.mp4"
    )

    visualize_detections(
        VIDEO_PATH,
        CSV_PATH,
        OUTPUT_VIDEO
    )