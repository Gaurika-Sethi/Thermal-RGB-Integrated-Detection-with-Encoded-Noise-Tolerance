import cv2


def draw_detections(image_path, detections):

    image = cv2.imread(image_path)

    for det in detections:

        x1, y1, x2, y2 = map(int, det["bbox"])

        conf = det["confidence"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            image,
            f"{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return image