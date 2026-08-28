from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app


class MockModelLoader:
    model_name = "CIFARClassifier"
    model_version = 7
    model = object()
    device = "cpu"

    def predict(self, tensor):
        return torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


@asynccontextmanager
async def noop_lifespan(app):
    yield


@pytest.fixture
def client():
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = noop_lifespan
    app.state.model_loader = MockModelLoader()

    with TestClient(app) as test_client:
        yield test_client

    app.router.lifespan_context = original_lifespan


@pytest.fixture
def getfile():
    def _getfile(filename: str) -> Path:
        return Path(f"tests/assets/{filename}")

    return _getfile


@pytest.fixture
def asset_file(request, getfile):
    filename, mime = request.param
    path = getfile(filename)

    with path.open("rb") as file:
        content = file.read()

    return (
        path.name,
        content,
        mime,
    )
