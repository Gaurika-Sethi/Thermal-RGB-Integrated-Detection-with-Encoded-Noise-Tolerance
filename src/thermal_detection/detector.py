import cv2
import matplotlib.pyplot as plt


def inspect_video(video_path):
    """
    Print pixel intensity statistics for every frame.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

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


def otsu_threshold_demo(video_path, num_frames=5):
    """
    Display original and thresholded thermal frames.
    """

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

        # Original
        plt.subplot(2, num_frames, i + 1)
        plt.imshow(gray, cmap="gray")
        plt.title(f"Original\nFrame {frame_no}")
        plt.axis("off")

        # Threshold
        plt.subplot(2, num_frames, num_frames + i + 1)
        plt.imshow(binary, cmap="gray")
        plt.title("Otsu Threshold")
        plt.axis("off")

    cap.release()

    plt.tight_layout()
    plt.show()


def contour_detection_demo(video_path, num_frames=5):
    """
    Detect contours and highlight the largest valid blob.
    """

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

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        valid_contours = []

        for contour in contours:

            area = cv2.contourArea(contour)

            # Uncomment this once if you want to inspect contour areas
            # print(f"Frame {frame_no} : Area = {area}")

            if 500 < area < 10000:
                valid_contours.append(contour)

        if valid_contours:

            largest = max(valid_contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest)

            cx = x + w // 2
            cy = y + h // 2

            cv2.drawContours(
                output,
                [largest],
                -1,
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            cv2.circle(
                output,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                output,
                f"({cx}, {cy})",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        plt.subplot(1, num_frames, i + 1)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_no}")
        plt.axis("off")

    cap.release()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    video_path = "../../data/raw/thermal_sample.mp4"

    # Uncomment whichever demo you want to run

    # inspect_video(video_path)

    # otsu_threshold_demo(video_path)

    contour_detection_demo(video_path)