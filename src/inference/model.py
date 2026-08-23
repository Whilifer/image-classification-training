import mlflow.pytorch
import torch


class ModelLoader:
    def __init__(
        self,
        model_name: str,
        model_version: int,
        device: torch.device,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.device = device

        self.model = None

    def load(self):
        model_uri = f"models:/{self.model_name}/{self.model_version}"

        print("Loading model:", model_uri)
        print("Device:", self.device)

        self.model = mlflow.pytorch.load_model(
            model_uri,
            map_location=self.device,
        )

        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        print("Model loaded")

    def predict(self, tensor: torch.Tensor):
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        tensor = tensor.to(
            self.device,
            non_blocking=True,
        )

        with torch.inference_mode():
            return self.model(tensor)
