import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.training.checkpoint import save_checkpoint
from src.training.evaluate import evaluate
from src.training.train import train_one_epoch

logger = logging.getLogger(__name__)


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float
    learning_rate: float


@dataclass
class TrainingResult:
    best_validation_accuracy: float
    best_epoch: int
    epochs_completed: int


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    early_stopping_enabled: bool,
    early_stopping_patience: int,
    checkpoint_path: str | Path,
    scheduler: LRScheduler | None = None,
    on_epoch_end: Callable[[EpochResult], None] | None = None,
) -> TrainingResult:
    best_validation_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_completed = 0

    for epoch in range(epochs):
        epochs_completed = epoch + 1

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

        current_learning_rate = optimizer.param_groups[0]["lr"]

        epoch_result = EpochResult(
            epoch=epoch + 1,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            learning_rate=current_learning_rate,
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch_result)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | validation_loss=%.4f | "
            "validation_accuracy=%.4f | learning_rate=%.8f",
            epoch + 1,
            epochs,
            train_loss,
            validation_loss,
            validation_accuracy,
            current_learning_rate,
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                path=checkpoint_path,
            )

            logger.info(
                "New best model saved: validation_accuracy=%.4f",
                best_validation_accuracy,
            )

        else:
            epochs_without_improvement += 1

        if (
            early_stopping_enabled
            and epochs_without_improvement >= early_stopping_patience
        ):
            logger.info(
                "Early stopping triggered after %d epochs without improvement",
                epochs_without_improvement,
            )
            break

        if scheduler is not None:
            scheduler.step()

    return TrainingResult(
        best_validation_accuracy=best_validation_accuracy,
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
    )
