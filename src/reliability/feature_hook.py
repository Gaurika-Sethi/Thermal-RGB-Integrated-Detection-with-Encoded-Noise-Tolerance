class FeatureHook:
    """
    Placeholder for future entropy and feature extraction hooks.
    Used for capturing intermediate detector activations.
    """

    def __init__(self):
        self.features = None

    def attach(self, model):
        """
        Attach forward hooks to target layers.
        """
        pass

    def extract(self):
        """
        Return extracted features.
        """
        return self.features