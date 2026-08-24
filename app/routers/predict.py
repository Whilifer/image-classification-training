from io import BytesIO

import torch
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.dependencies import get_model_loader
from app.schemas import PredictionResponse
from src.data.dataset import CLASSES
from src.inference.model import ModelLoader
from src.inference.preprocessing import preprocess

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post(
    "/",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
    model_loader: ModelLoader = Depends(get_model_loader),
):
    if not file.content_type or not file.content_type.startswith("image/"):
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

    class_id = index.item()

    return PredictionResponse(
        class_id=class_id,
        class_name=CLASSES[class_id],
        confidence=round(
            confidence.item(),
            4,
        ),
    )
