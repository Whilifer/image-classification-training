import logging

import optuna
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import TrainConfig
from src.data.dataset import create_dataloaders
from src.models.classifier import CIFARClassifier
from src.training.pipeline import EpochResult, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial) -> float:
    config = TrainConfig.from_yaml("configs/train.yaml")

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        3e-3,
        log=True,
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-3,
        log=True,
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [64, 128, 256],
    )

    logger.info(
        "Trial %d | learning_rate=%.6f | weight_decay=%.6f | batch_size=%d",
        trial.number,
        learning_rate,
        weight_decay,
        batch_size,
    )

    device = config.get_device()

    train_loader, validation_loader, _ = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=batch_size,
        num_workers=config.num_workers,
        augmentation=config.augmentation,
    )

    model = CIFARClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = None

    if config.scheduler.enabled:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.scheduler.min_learning_rate,
        )

    def on_epoch_end(result: EpochResult) -> bool:
        trial.report(
            result.validation_accuracy,
            step=result.epoch,
        )

        if trial.should_prune():
            logger.info(
                "Trial %d pruned at epoch %d",
                trial.number,
                result.epoch,
            )
            return True

        return False

    training_result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config.epochs,
        early_stopping_enabled=config.early_stopping.enabled,
        early_stopping_patience=config.early_stopping.patience,
        checkpoint_path=f"artifacts/optuna_trial_{trial.number}.pt",
        scheduler=scheduler,
        on_epoch_end=on_epoch_end,
    )

    if trial.should_prune():
        raise optuna.TrialPruned()

    validation_accuracy = training_result.best_validation_accuracy

    logger.info(
        "Trial %d finished | best_validation_accuracy=%.4f | best_epoch=%d",
        trial.number,
        validation_accuracy,
        training_result.best_epoch,
    )

    return validation_accuracy


def main() -> None:
    study = optuna.create_study(
        study_name="cifar10_hyperparameter_optimization",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
        ),
    )

    study.optimize(
        objective,
        n_trials=10,
    )

    print()
    print("Best trial:")
    print(f"Value: {study.best_value}")
    print(f"Params: {study.best_params}")


if __name__ == "__main__":
    main()
