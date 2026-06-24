import cv2
import os


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count
    }


def extract_frames(video_path, output_folder):
    output_folder = "../../data/processed/extracted_frames"

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_name = os.path.join(
            output_folder,
            f"frame_{frame_idx:04d}.jpg"
        )

        cv2.imwrite(frame_name, frame)

        frame_idx += 1

    cap.release()

    print(f"Saved {frame_idx} frames")

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_name = os.path.join(
            output_folder,
            f"frame_{frame_idx:04d}.jpg"
        )

        cv2.imwrite(frame_name, frame)

        frame_idx += 1

    cap.release()

    print(f"Saved {frame_idx} frames")


def create_video_from_frames(
        frames_folder,
        output_video_path,
        fps):
    
    output_video_path = "../../outputs/reconstructed.mp4"

    frame_files = sorted(
        [
            f for f in os.listdir(frames_folder)
            if f.endswith(".jpg")
        ]
    )

    first_frame = cv2.imread(
        os.path.join(frames_folder, frame_files[0])
    )

    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    for frame_file in frame_files:

        frame = cv2.imread(
            os.path.join(frames_folder, frame_file)
        )

        writer.write(frame)

    writer.release()

    print("Video created:", output_video_path)