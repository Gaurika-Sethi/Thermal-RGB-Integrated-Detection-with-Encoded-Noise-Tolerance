from video_utils import (
    get_video_info,
    extract_frames,
    create_video_from_frames
)

video_path = "../../data/raw/sample.mp4"

info = get_video_info(video_path)

extract_frames(
    video_path,
    "outputs/extracted_frames"
)

create_video_from_frames(
    "data/processed/extracted_frames",
    "outputs/reconstructed.mp4",
    info["fps"]
)