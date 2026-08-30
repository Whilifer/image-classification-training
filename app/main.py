from fastapi import FastAPI

from app.lifespan import lifespan
from app.routers.health import router as health_router
from app.routers.predict import router as predict_router

app = FastAPI(
    title="CIFAR-10 Classifier",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(predict_router)
