import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.evaluate import evaluate


def test_evaluate_does_not_change_model():
    model = nn.Linear(4, 2)

    images = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,))

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4)

    criterion = nn.CrossEntropyLoss()

    before = [parameter.detach().clone() for parameter in model.parameters()]

    evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    after = list(model.parameters())

    for before_parameter, after_parameter in zip(
        before,
        after,
        strict=True,
    ):
        assert torch.equal(
            before_parameter,
            after_parameter,
        )


def test_evaluate_accuracy():
    model = nn.Linear(2, 2)

    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([1.0, 0.0]))

    images = torch.randn(4, 2)
    labels = torch.tensor([0, 0, 1, 0])

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
    )

    criterion = nn.CrossEntropyLoss()

    loss, accuracy = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert accuracy == 0.75
    assert loss > 0


def test_evaluate_sets_eval_mode():
    model = nn.Linear(2, 2)
    model.train()

    images = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 0, 1])

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
    )

    criterion = nn.CrossEntropyLoss()

    assert model.training is True

    evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert model.training is False
