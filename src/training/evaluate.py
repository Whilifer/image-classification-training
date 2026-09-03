import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            batch_size = images.size(0)

            total_loss += loss * batch_size

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total_samples += batch_size

    average_loss = total_loss.item() / total_samples
    accuracy = correct / total_samples

    return average_loss, accuracy


def evaluate_exported_model(
    model,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()

        total_samples += batch_size

    return (
        total_loss / total_samples,
        correct / total_samples,
    )


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()

    targets = []
    predictions = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            targets.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())

    return targets, predictions
