from pydantic import BaseModel


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str
    model_version: int
    device: str
