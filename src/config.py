from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass
class TrainConfig:
    data_dir: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    device: str
    mlflow_tracking_uri: str

    augmentation: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        path = Path(path)

        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        return cls(**config)

    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return torch.device(self.device)
