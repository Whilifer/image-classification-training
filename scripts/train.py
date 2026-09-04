import logging

import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from torch import inference_mode, nn, testing
from torch.optim import Adam

from logging_config import setup_logging
from src.config import TrainConfig
from src.data.dataset import create_dataloaders
from src.models.classifier import CIFARClassifier
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.evaluate import collect_predictions, evaluate
from src.training.metrics import classification_metrics
from src.training.train import train_one_epoch


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    config = TrainConfig.from_yaml("configs/train.yaml")

    device = config.get_device()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")

    train_loader, validation_loader, test_loader = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augmentation=config.augmentation,
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
    epochs_without_improvement = 0

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    # CIFAR10-classification  CIFAR10-classification-docker
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
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
                "augmentation_enabled": config.augmentation.enabled,
                "augmentation_horizontal_flip": config.augmentation.horizontal_flip,
                "augmentation_random_crop": config.augmentation.random_crop,
                "augmentation_crop_padding": config.augmentation.crop_padding,
                "augmentation_random_rotation": config.augmentation.random_rotation,
                "augmentation_rotation_degrees": config.augmentation.rotation_degrees,
                "early_stopping_enabled": config.early_stopping.enabled,
                "early_stopping_patience": config.early_stopping.patience,
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

            logger.info(f"Epoch {epoch + 1}/{config.epochs}")
            logger.info(f"Train loss: {train_loss:.4f}")
            logger.info(f"Validation loss: {validation_loss:.4f}")
            logger.info(f"Validation accuracy: {validation_accuracy:.4f}")

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                save_checkpoint(
                    model=model,
                    path="artifacts/best_model.pt",
                )

                logger.info("New best model saved")
            else:
                epochs_without_improvement += 1

            if (
                config.early_stopping.enabled
                and epochs_without_improvement >= config.early_stopping.patience
            ):
                logger.info(
                    "Early stopping triggered after %d epochs without improvement",
                    epochs_without_improvement,
                )
                break

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

        logger.info(f"Model URI: {model_uri}")

        loaded_model = mlflow.pytorch.load_model(model_uri)
        # loaded_model.eval()

        with inference_mode():
            loaded_output = loaded_model(example_images_cpu)

        testing.assert_close(
            example_outputs_cpu,
            loaded_output,
        )

        logger.info("MLflow model successfully loaded and verified")

        model.to(device)

        test_loss, test_accuracy = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        targets, predictions = collect_predictions(
            model=model,
            dataloader=test_loader,
            device=device,
        )

        metrics = classification_metrics(
            targets=targets,
            predictions=predictions,
        )

        mlflow.log_metrics(
            {
                "test_accuracy": metrics["accuracy"],
                "test_precision_macro": metrics["precision_macro"],
                "test_recall_macro": metrics["recall_macro"],
                "test_f1_macro": metrics["f1_macro"],
            }
        )

        mlflow.log_dict(
            metrics["confusion_matrix"],
            "metrics/confusion_matrix.json",
        )

        mlflow.log_metric(
            "best_validation_accuracy",
            best_validation_accuracy,
        )

        mlflow.log_metric(
            "best_epoch",
            best_epoch,
        )

        mlflow.log_metric(
            "epochs_completed",
            best_epoch if config.early_stopping.enabled else config.epochs,
        )

        mlflow.log_artifact("artifacts/best_model.pt")
        mlflow.log_artifact(
            "configs/train.yaml",
        )

        logger.info(f"Final test Loss: {test_loss:.4f}")
        logger.info(f"Final test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
