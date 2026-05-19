import torch


class FeatureExtractor:

    def __init__(self, model):

        self.model = model

        self.features = {}

        self.hooks = []


    def hook_fn(self, layer_name):

        def fn(module, input, output):

            # Handle tuple outputs
            if isinstance(output, tuple):

                processed = []

                for item in output:

                    if torch.is_tensor(item):
                        processed.append(
                            item.detach().cpu()
                        )

                self.features[layer_name] = processed


            # Handle tensor outputs
            elif torch.is_tensor(output):

                self.features[layer_name] = (
                    output.detach().cpu()
                )

        return fn


    def register_hooks(self):

        """
        Attach hooks to YOLO layers
        """

        for idx, layer in enumerate(self.model.model.model):

            layer_name = f"layer_{idx}"

            hook = layer.register_forward_hook(
                self.hook_fn(layer_name)
            )

            self.hooks.append(hook)


    def remove_hooks(self):

        for hook in self.hooks:
            hook.remove()

        self.hooks = []


    def get_features(self):

        return self.features