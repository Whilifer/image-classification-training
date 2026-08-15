from src.data.dataset import create_dataloaders


def test_dataset_split_sizes():
    train_loader, validation_loader, test_loader = create_dataloaders(
        data_dir="data",
        batch_size=128,
        num_workers=0,
    )

    assert len(train_loader.dataset) == 45000
    assert len(validation_loader.dataset) == 5000
    assert len(test_loader.dataset) == 10000


def test_dataset_batch_shapes():
    train_loader, validation_loader, test_loader = create_dataloaders(
        data_dir="data",
        batch_size=128,
        num_workers=0,
    )

    train_images, train_labels = next(iter(train_loader))
    validation_images, validation_labels = next(iter(validation_loader))
    test_images, test_labels = next(iter(test_loader))

    assert train_images.shape == (128, 3, 32, 32)
    assert train_labels.shape == (128,)

    assert validation_images.shape == (128, 3, 32, 32)
    assert validation_labels.shape == (128,)

    assert test_images.shape == (128, 3, 32, 32)
    assert test_labels.shape == (128,)
