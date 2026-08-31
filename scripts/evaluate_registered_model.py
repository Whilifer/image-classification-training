import logging

import mlflow
import torch
from torch import nn

from src.data.dataset import create_dataloaders
from src.training.evaluate import evaluate_exported_model

logger = logging.getLogger(__name__)

MODEL_URI = "models:/CIFARClassifier/2"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")
    logger.info(f"Loading model: {MODEL_URI}")

    model = mlflow.pytorch.load_model(
        MODEL_URI,
        map_location=device,
    )

    model = model.to(device)

    _, _, test_loader = create_dataloaders(
        data_dir="data",
        batch_size=128,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate_exported_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    logger.info(
        f"Registered model | Test loss: {test_loss:.4f} | Accuracy: {test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
