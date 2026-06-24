from video_utils import get_video_info

video_path = "../../data/raw/sample.mp4"

info = get_video_info(video_path)

print(info)

from video_utils import extract_frames

extract_frames(
    video_path,
    "data/processed/extracted_frames"
)