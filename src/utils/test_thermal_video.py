from video_utils import get_video_info

video_path = "../../data/raw/thermal_sample.mp4"

info = get_video_info(video_path)

print(info)