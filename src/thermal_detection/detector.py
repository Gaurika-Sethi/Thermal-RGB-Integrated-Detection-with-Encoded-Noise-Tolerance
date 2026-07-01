import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------- Tunable Parameters ----------------------

MIN_BLOB_AREA = 300
MAX_BLOB_AREA = 20000

EXPECTED_ASPECT_RATIO = 2.5
MAX_EXPECTED_AREA = 15000

WEIGHT_CONTRAST = 0.40
WEIGHT_INTENSITY = 0.25
WEIGHT_AREA = 0.20
WEIGHT_ASPECT = 0.15

LOCAL_CONTRAST_BLUR = (15, 15)   # kernel for background estimation
LOCAL_CONTRAST_THRESHOLD = 10    # minimum local heat above surroundings


# ---------------------- Shared Helpers ----------------------

def apply_threshold(gray):
    """
    Local contrast thresholding.
    Subtracts a blurred background estimate so that only pixels
    locally hotter than their surroundings survive.
    Replaces global Otsu which was picking up the bright floor.
    """
    blurred = cv2.GaussianBlur(gray, LOCAL_CONTRAST_BLUR, 0)
    diff = cv2.subtract(gray, blurred)
    _, binary = cv2.threshold(diff, LOCAL_CONTRAST_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def compute_confidence(c, gray):
    """
    Returns a (confidence, area_score, intensity_score, contrast_score, aspect_score)
    tuple for a single contour.
    All components are logged separately so they can be inspected during tuning.
    """
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)

    blob_pixels = gray[mask == 255]
    background_pixels = gray[mask == 0]

    if not len(blob_pixels):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    mean_blob = np.mean(blob_pixels)
    mean_background = np.mean(background_pixels)

    area_score      = min(area / MAX_EXPECTED_AREA, 1.0)
    intensity_score = mean_blob / 255.0
    contrast_score  = max((mean_blob - mean_background) / 255.0, 0.0)
    aspect          = h / w if w else 0
    aspect_score    = max(
        1 - abs(aspect - EXPECTED_ASPECT_RATIO) / EXPECTED_ASPECT_RATIO, 0.0
    )

    confidence = (
        WEIGHT_CONTRAST  * contrast_score  +
        WEIGHT_INTENSITY * intensity_score +
        WEIGHT_AREA      * area_score      +
        WEIGHT_ASPECT    * aspect_score
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return confidence, area_score, intensity_score, contrast_score, aspect_score


def get_valid_contours(contours):
    """Filter contours by area bounds."""
    return [
        c for c in contours
        if MIN_BLOB_AREA < cv2.contourArea(c) < MAX_BLOB_AREA
    ]


def best_contour(valid_contours, gray):
    """Pick the contour with the highest confidence score."""
    return max(valid_contours, key=lambda c: compute_confidence(c, gray)[0])


# ---------------------- Day 9 ----------------------

def inspect_video(video_path):
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


# ---------------------- Day 10 ----------------------

def otsu_threshold_demo(video_path, num_frames=5):
    """
    Kept for reference — shows original frame alongside the
    local-contrast binary so you can see what changed from Otsu.
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

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = apply_threshold(gray)

        plt.subplot(2, num_frames, i + 1)
        plt.imshow(gray, cmap="gray")
        plt.title(f"Original\nFrame {frame_no}")
        plt.axis("off")

        plt.subplot(2, num_frames, num_frames + i + 1)
        plt.imshow(binary, cmap="gray")
        plt.title("Local Contrast Binary")
        plt.axis("off")

    cap.release()
    plt.tight_layout()
    plt.show()


# ---------------------- Day 11 ----------------------

def contour_detection_demo(video_path, num_frames=5):
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

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = apply_threshold(gray)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        output         = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        valid_contours = get_valid_contours(contours)

        if valid_contours:
            chosen = best_contour(valid_contours, gray)
            x, y, w, h = cv2.boundingRect(chosen)
            cx, cy = x + w // 2, y + h // 2

            cv2.drawContours(output, [chosen], -1, (0, 255, 0), 2)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                output, f"({cx},{cy})",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        plt.subplot(1, num_frames, i + 1)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_no}")
        plt.axis("off")

    cap.release()
    plt.tight_layout()
    plt.show()


# ---------------------- Day 12 & 13 ----------------------

def confidence_demo(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    scores = []
    plt.figure(figsize=(15, 10))

    for i in range(num_frames):
        frame_no = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            continue

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = apply_threshold(gray)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # debug prints — both after contours and valid_contours are assigned
        all_areas      = sorted([cv2.contourArea(c) for c in contours], reverse=True)
        valid_contours = get_valid_contours(contours)
        print(f"Frame {frame_no} — all contour areas (top 10): {all_areas[:10]}")
        print(f"Frame {frame_no} — valid after area filter: {len(valid_contours)}")

        output     = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        confidence = 0.0

        if valid_contours:
            chosen = best_contour(valid_contours, gray)
            confidence, area_score, intensity_score, contrast_score, aspect_score = \
                compute_confidence(chosen, gray)

            x, y, w, h = cv2.boundingRect(chosen)
            cx, cy = x + w // 2, y + h // 2

            cv2.drawContours(output, [chosen], -1, (0, 255, 0), 2)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                output, f"Conf:{confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            scores.append({
                "frame":           frame_no,
                "cx":              cx,
                "cy":              cy,
                "area_score":      round(area_score,      4),
                "intensity_score": round(intensity_score, 4),
                "contrast_score":  round(contrast_score,  4),
                "aspect_score":    round(aspect_score,    4),
                "confidence":      round(confidence,      4),
            })

        plt.subplot(1, num_frames, i + 1)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_no}\nConf={confidence:.2f}")
        plt.axis("off")

    cap.release()
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(scores)
    out_path = "../../data/processed/thermal_confidence_scores.csv"
    df.to_csv(out_path, index=False)
    print("\n--- Confidence scores ---")
    print(df.to_string(index=False))


if __name__ == "__main__":

    video_path = "../../data/raw/thermal_sample.mp4"

    # inspect_video(video_path)
    # otsu_threshold_demo(video_path)
    # contour_detection_demo(video_path)
    confidence_demo(video_path)