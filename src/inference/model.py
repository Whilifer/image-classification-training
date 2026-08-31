import logging

import mlflow
import mlflow.pytorch
import torch

logger = logging.getLogger(__name__)


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

        logger.info("1. Starting model load")
        logger.info(f"Model URI:, {model_uri}")
        logger.info(f"Tracking URI:, {self.mlflow_tracking_uri}")
        logger.info(f"Device:, {self.device}")

        logger.info("2. Calling mlflow.pytorch.load_model()")

        try:
            self.model = mlflow.pytorch.load_model(
                model_uri,
                map_location=self.device,
            )
        except Exception:
            logger.exception(f"Failed to load model: %s, {model_uri}")
            raise

        logger.info("3. MLflow model loaded")

        if hasattr(self.model, "to"):
            logger.info("4. Moving model to device")
            self.model = self.model.to(self.device)

        logger.info("5. Model loaded successfully")

    def predict(self, tensor: torch.Tensor):
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        tensor = tensor.to(
            self.device,
            non_blocking=True,
        )

        with torch.inference_mode():
            return self.model(tensor)
