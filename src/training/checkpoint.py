from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
) -> None:
    path = Path(path)

    state_dict = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(state_dict)
