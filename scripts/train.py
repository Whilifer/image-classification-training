import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from torch import inference_mode, nn, testing
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
    best_epoch = 0

    mlflow.set_experiment("CIFAR10-classification")

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    with mlflow.start_run(run_name="baseline"):
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

        mlflow.set_tags(
            {
                "project": "cifar10-classification",
                "dataset": "CIFAR10",
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
                best_epoch = epoch + 1

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

        example_images, _ = next(iter(test_loader))
        example_images = example_images.to(device)

        model_cpu = model.to("cpu")
        example_images_cpu = example_images.cpu()

        with inference_mode():
            example_outputs_cpu = model_cpu(example_images_cpu)

        signature = infer_signature(
            example_images_cpu.numpy(), example_outputs_cpu.numpy()
        )

        mlflow.pytorch.log_model(
            model_cpu,
            name="model",
            signature=signature,
            serialization_format="pt2",
            input_example=example_images_cpu.numpy(),
            registered_model_name="CIFARClassifier",
        )

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

        print("Model URI:", model_uri)

        loaded_model = mlflow.pytorch.load_model(model_uri)
        # loaded_model.eval()

        with inference_mode():
            loaded_output = loaded_model(example_images_cpu)

        testing.assert_close(
            example_outputs_cpu,
            loaded_output,
        )

        print("MLflow model successfully loaded and verified")

        model.to(device)

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

        mlflow.log_metric(
            "best_validation_accuracy",
            best_validation_accuracy,
        )

        mlflow.log_metric(
            "best_epoch",
            best_epoch,
        )

        mlflow.log_artifact("artifacts/best_model.pt")
        mlflow.log_artifact(
            "configs/train.yaml",
        )

        print(f"Final test | Loss: {test_loss:.4f} | Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
