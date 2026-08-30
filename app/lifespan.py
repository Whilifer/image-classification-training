import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.inference.config import load_config, resolve_device
from src.inference.model import ModelLoader

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config("configs/inference.yaml")

    device = resolve_device(config.device)

    model_loader = ModelLoader(
        model_name=config.model_name,
        model_version=config.model_version,
        device=device,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
    )

    logger.info("Starting model load")

    model_loader.load()

    app.state.model_loader = model_loader

    logger.info("Model ready")

    yield

    app.state.model_loader = None
