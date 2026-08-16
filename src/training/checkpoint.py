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
