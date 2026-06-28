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