from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import AugmentationConfig

CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def create_train_transform(
    augmentation: AugmentationConfig,
):
    transforms_list = []

    if augmentation.horizontal_flip:
        transforms_list.append(transforms.RandomHorizontalFlip())

    if augmentation.random_crop:
        transforms_list.append(
            transforms.RandomCrop(
                32,
                padding=augmentation.crop_padding,
            )
        )

    if augmentation.random_rotation:
        transforms_list.append(
            transforms.RandomRotation(
                augmentation.rotation_degrees,
            )
        )

    transforms_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    return transforms.Compose(transforms_list)


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    augmentation: AugmentationConfig | None = None,
):
    data_dir = Path(data_dir)

    if augmentation is not None and augmentation.enabled:
        train_transform = create_train_transform(augmentation)
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    validation_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )

    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        len(train_dataset),
        generator=generator,
    )

    train_indices = indices[:45_000]
    validation_indices = indices[45_000:]

    train_dataset = Subset(
        train_dataset,
        train_indices,
    )

    validation_dataset = Subset(
        validation_dataset,
        validation_indices,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, validation_loader, test_loader
