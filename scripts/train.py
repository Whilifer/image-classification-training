import time

import torch
from torch import nn
from torch.optim import Adam

from src.data.dataset import create_dataloaders
from src.models.classifier import CIFARClassifier
from src.training.train import train_one_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

    train_loader, _ = create_dataloaders(
        data_dir="data",
        batch_size=128,
        num_workers=0,
    )

    print("Train batches:", len(train_loader))

    model = CIFARClassifier().to(device)

    print("Model device:", next(model.parameters()).device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.001,
    )

    print("Checking first batch...")

    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    print("Images before:", images.device)
    print("Labels before:", labels.device)

    images = images.to(device)
    labels = labels.to(device)

    print("Images after:", images.device)
    print("Labels after:", labels.device)

    print("Benchmarking one training step...")

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    elapsed = time.perf_counter() - start

    print(f"One training step: {elapsed:.4f} sec")
    print(f"Loss: {loss.item():.4f}")

    print("Starting full epoch...")

    epoch_start = time.perf_counter()

    train_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    epoch_elapsed = time.perf_counter() - epoch_start

    print(f"Train loss: {train_loss:.4f}")
    print(f"Epoch time: {epoch_elapsed:.2f} sec")


if __name__ == "__main__":
    main()
