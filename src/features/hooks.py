from ultralytics import YOLO


class FeatureExtractor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.features = None
        self.hook = None

    def _hook_fn(self, module, input, output):
        """
        Stores feature maps during forward pass.
        """
        self.features = output.detach()

    def register_hook(self, layer_index=15):
        """
        Register forward hook on selected YOLO layer.
        """

        layer = self.model.model.model[layer_index]

        self.hook = layer.register_forward_hook(self._hook_fn)

        print(f"Hook registered on layer {layer_index}")

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()
            print("Hook removed")