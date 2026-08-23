from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers.health import router as health_router
from src.inference.config import load_config, resolve_device
from src.inference.model import ModelLoader


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config("configs/inference.yaml")
    device = resolve_device(config.device)

    model_loader = ModelLoader(
        model_name=config.model_name,
        model_version=config.model_version,
        device=device,
    )

    model_loader.load()

    app.state.model_loader = model_loader

    yield

    app.state.model_loader = None


app = FastAPI(
    title="CIFAR-10 Classifier",
    lifespan=lifespan,
)

app.include_router(health_router)
