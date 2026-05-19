class DetectionRouter:

    def __init__(self, rgb_detector, thermal_detector):

        self.rgb_detector = rgb_detector

        self.thermal_detector = thermal_detector


    def route(
        self,
        image_path,
        modality
    ):

        modality = modality.lower()

        if modality == "rgb":

            return self.rgb_detector.predict(image_path)

        elif modality == "thermal":

            return self.thermal_detector.predict(image_path)

        else:

            raise ValueError(
                f"Unsupported modality: {modality}"
            )