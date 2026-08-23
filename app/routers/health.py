from fastapi import APIRouter, Request

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("/")
def health(request: Request):
    model_loader = request.app.state.model_loader

    return {
        "status": "ok",
        "model_loaded": model_loader.model is not None,
        "model": model_loader.model_name,
        "model_version": model_loader.model_version,
        "device": str(model_loader.device),
    }
