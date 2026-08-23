from io import BytesIO

import torch
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.dependencies import get_model_loader
from src.inference.model import ModelLoader

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post("/")
async def predict(
    file: UploadFile = File(...),
    model_loader: ModelLoader = Depends(get_model_loader),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image",
        ) from None

    tensor = preprocess(image)

    output = model_loader.predict(tensor)

    probabilities = torch.softmax(
        output,
        dim=1,
    )

    confidence, index = probabilities[0].max(dim=0)

    return {
        "class_id": index.item(),
        "confidence": round(
            confidence.item(),
            4,
        ),
    }


def preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    tensor = transform(image)

    return tensor.unsqueeze(0)
