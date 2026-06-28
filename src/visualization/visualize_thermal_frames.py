import cv2
import matplotlib.pyplot as plt


def show_sample_frames(video_path, num_frames=5):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    plt.figure(figsize=(18, 5))

    for i in range(num_frames):

        frame_no = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        plt.subplot(1, num_frames, i + 1)

        plt.imshow(gray, cmap="gray")

        plt.title(
            f"Frame {frame_no}\n"
            f"Min:{gray.min()} Max:{gray.max()}\n"
            f"Mean:{gray.mean():.1f}"
        )

        plt.axis("off")

    cap.release()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    video_path = "../../data/raw/thermal_sample.mp4"

    show_sample_frames(video_path, num_frames=5)