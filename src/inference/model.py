import mlflow
import mlflow.pytorch
import torch


class ModelLoader:
    def __init__(
        self,
        model_name: str,
        model_version: int,
        device: torch.device,
        mlflow_tracking_uri: str,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.device = device
        self.mlflow_tracking_uri = mlflow_tracking_uri

        self.model = None

    def load(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        model_uri = f"models:/{self.model_name}/{self.model_version}"

        print("1. Starting model load")
        print("Model URI:", model_uri)
        print("Tracking URI:", self.mlflow_tracking_uri)
        print("Device:", self.device, flush=True)

        print("2. Calling mlflow.pytorch.load_model()", flush=True)

        self.model = mlflow.pytorch.load_model(
            model_uri,
            map_location=self.device,
        )

        print("3. MLflow model loaded", flush=True)

        if hasattr(self.model, "to"):
            print("4. Moving model to device", flush=True)
            self.model = self.model.to(self.device)

        print("5. Model loaded successfully", flush=True)

    def predict(self, tensor: torch.Tensor):
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        tensor = tensor.to(
            self.device,
            non_blocking=True,
        )

        with torch.inference_mode():
            return self.model(tensor)
