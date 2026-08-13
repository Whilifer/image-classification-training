import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = torch.tensor(0.0, device=device)
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss * batch_size
        total_samples += batch_size

    return total_loss.item() / total_samples
