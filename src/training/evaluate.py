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
