from fastapi import APIRouter, Request

from app.schemas import HealthResponse

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "/",
    response_model=HealthResponse,
)
def health(request: Request):
    model_loader = request.app.state.model_loader

    return HealthResponse(
        status="ok",
        model_loaded=model_loader.model is not None,
        model=model_loader.model_name,
        model_alias=model_loader.model_alias,
        device=str(model_loader.device),
    )
