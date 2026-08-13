import torch

from src.models.classifier import CIFARClassifier


def test_model_output_shape():
    model = CIFARClassifier()

    images = torch.randn(8, 3, 32, 32)

    output = model(images)

    assert output.shape == (8, 10)


def test_model_output_is_finite():
    model = CIFARClassifier()

    images = torch.randn(8, 3, 32, 32)

    output = model(images)

    assert torch.isfinite(output).all()
