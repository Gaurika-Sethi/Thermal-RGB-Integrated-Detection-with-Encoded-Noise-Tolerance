import cv2


def inspect_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print(
            f"Frame {frame_number}: "
            f"Min={gray.min()} "
            f"Max={gray.max()} "
            f"Mean={gray.mean():.2f}"
        )

        frame_number += 1

    cap.release()


inspect_video("../../data/raw/thermal_sample.mp4")

import cv2
import matplotlib.pyplot as plt


def otsu_threshold_demo(video_path, num_frames=5):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    plt.figure(figsize=(15, 10))

    for i in range(num_frames):

        frame_no = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Original frame
        plt.subplot(2, num_frames, i + 1)
        plt.imshow(gray, cmap="gray")
        plt.title(f"Original\nFrame {frame_no}")
        plt.axis("off")

        # Thresholded frame
        plt.subplot(2, num_frames, num_frames + i + 1)
        plt.imshow(binary, cmap="gray")
        plt.title("Otsu Threshold")
        plt.axis("off")

    cap.release()

    plt.tight_layout()
    plt.show()


video_path = "../../data/raw/thermal_sample.mp4"

otsu_threshold_demo(video_path)