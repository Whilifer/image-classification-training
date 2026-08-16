import mlflow
from torch import nn
from torch.optim import Adam

from src.config import TrainConfig
from src.data.dataset import create_dataloaders
from src.models.classifier import CIFARClassifier
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.evaluate import evaluate
from src.training.train import train_one_epoch


def main():
    config = TrainConfig.from_yaml("configs/train.yaml")

    device = config.get_device()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    train_loader, validation_loader, test_loader = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = CIFARClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_validation_accuracy = 0.0

    mlflow.set_experiment("CIFAR10-classification")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "weight_decay": config.weight_decay,
                "num_workers": config.num_workers,
                "device": str(device),
                "optimizer": "Adam",
                "model": "CIFARClassifier",
            }
        )
        for epoch in range(config.epochs):
            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            validation_loss, validation_accuracy = evaluate(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                },
                step=epoch + 1,
            )

            print(
                f"Epoch {epoch + 1}/{config.epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Validation loss: {validation_loss:.4f} | "
                f"Validation accuracy: {validation_accuracy:.4f}"
            )

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy

                save_checkpoint(
                    model=model,
                    path="artifacts/best_model.pt",
                )

                print("New best model saved")

        best_model_path = "artifacts/best_model.pt"

        load_checkpoint(
            model=model,
            path=best_model_path,
            device=device,
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

        mlflow.log_artifact("artifacts/best_model.pt")

        print(f"Final test | Loss: {test_loss:.4f} | Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
