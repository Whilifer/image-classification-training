from contextlib import asynccontextmanager

import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app


class MockModelLoader:
    model_name = "CIFARClassifier"
    model_alias = "champion"
    model = object()
    device = "cpu"

    def predict(self, tensor):
        assert tensor.shape == (1, 3, 32, 32)
        assert tensor.dtype == torch.float32

        return torch.tensor(
            [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )


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
