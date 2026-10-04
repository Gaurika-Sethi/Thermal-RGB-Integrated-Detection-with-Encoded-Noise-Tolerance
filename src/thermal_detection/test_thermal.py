from pathlib import Path
from detector import detect_people

project_root = Path(__file__).resolve().parents[2]

video_path = "C:/thermal project/Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance/data/raw/thermal_vid.mp4"

output_csv = "C:/thermal project/Thermal-RGB-Integrated-Detection-with-Encoded-Noise-Tolerance/outputs/thermal_multiple_people.csv"


detect_people(
    video_path,
    output_csv
)