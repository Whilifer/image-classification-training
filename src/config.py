from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass
class SchedulerConfig:
    enabled: bool = False
    type: str = "cosine"
    min_learning_rate: float = 0.00001


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 5


@dataclass
class AugmentationConfig:
    enabled: bool = False
    horizontal_flip: bool = False
    random_crop: bool = False
    crop_padding: int = 4
    random_rotation: bool = False
    rotation_degrees: float = 15.0


@dataclass
class TrainConfig:
    experiment_name: str
    run_name: str
    data_dir: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    device: str
    mlflow_tracking_uri: str
    augmentation: AugmentationConfig
    early_stopping: EarlyStoppingConfig
    scheduler: SchedulerConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        path = Path(path)

        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        config["augmentation"] = AugmentationConfig(**config.get("augmentation", {}))
        config["early_stopping"] = EarlyStoppingConfig(
            **config.get("early_stopping", {})
        )
        config["scheduler"] = SchedulerConfig(**config.get("scheduler", {}))

        return cls(**config)

    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return torch.device(self.device)
