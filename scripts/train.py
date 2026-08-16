import torch
from torch import nn
from torch.optim import Adam

from src.data.dataset import create_dataloaders
from src.models.classifier import CIFARClassifier
from src.training.checkpoint import save_checkpoint
from src.training.evaluate import evaluate
from src.training.train import train_one_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    train_loader, validation_loader, test_loader = create_dataloaders(
        data_dir="data",
        batch_size=128,
        num_workers=0,
    )

    model = CIFARClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 10

    best_validation_accuracy = 0.0

    for epoch in range(epochs):
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

        print(
            f"Epoch {epoch + 1}/{epochs} | "
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


if __name__ == "__main__":
    main()
