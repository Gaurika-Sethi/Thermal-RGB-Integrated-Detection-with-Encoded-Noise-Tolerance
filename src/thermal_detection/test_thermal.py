from pathlib import Path
from detector import detect_people

project_root = Path(__file__).resolve().parents[2]

video_path = project_root / "data" / "raw" / "thermal_sample.mp4"

output_csv = project_root / "outputs" / "thermal_detections.csv"

detect_people(
    video_path,
    output_csv
)