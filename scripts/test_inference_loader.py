import logging

import torch

from src.inference.config import (
    load_config,
    resolve_device,
)
from src.inference.model import ModelLoader

logger = logging.getLogger(__name__)


def main():
    config = load_config("configs/inference.yaml")

    device = resolve_device(config.device)

    loader = ModelLoader(
        model_name=config.model_name,
        model_version=config.model_version,
        device=device,
    )

    loader.load()

    example = torch.randn(
        1,
        3,
        32,
        32,
        device=device,
    )

    output = loader.predict(example)

    logger.info(f"Input shape: {example.shape}")
    logger.info(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
