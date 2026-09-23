import logging

import mlflow
import mlflow.pytorch
from torch import nn

from logging_config import setup_logging
from src.config import TrainConfig
from src.data.dataset import create_dataloaders
from src.training.evaluate import evaluate_exported_model

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    config = TrainConfig.from_yaml("configs/train.yaml")

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    device = config.get_device()

    model_uri = "models:/CIFARClassifier@champion"

    logger.info("Device: %s", device)
    logger.info("MLflow tracking URI: %s", config.mlflow_tracking_uri)
    logger.info("Loading registered model: %s", model_uri)

    model = mlflow.pytorch.load_model(
        model_uri,
        map_location=device,
    )

    model = model.to(device)

    _, _, test_loader = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate_exported_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    logger.info(
        "Registered model | Test loss: %.4f | Accuracy: %.4f",
        test_loss,
        test_accuracy,
    )


if __name__ == "__main__":
    main()
