from fastapi import Request

from src.inference.model import ModelLoader


def get_model_loader(
    request: Request,
) -> ModelLoader:
    return request.app.state.model_loader
