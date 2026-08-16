import torch
from torch import nn

from src.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_save_and_load(tmp_path):
    model = nn.Linear(4, 2)

    original_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }

    path = tmp_path / "model.pt"

    save_checkpoint(
        model=model,
        path=path,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    load_checkpoint(
        model=model,
        path=path,
        device=torch.device("cpu"),
    )

    for key, parameter in model.state_dict().items():
        assert torch.equal(
            parameter,
            original_state[key],
        )
