from pathlib import Path

import torch
import yaml
from pydantic import BaseModel


class InferenceConfig(BaseModel):
    model_name: str
    model_alias: str
    device: str
    mlflow_tracking_uri: str


def load_config(
    path: str | Path,
) -> InferenceConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return InferenceConfig(**data)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)
