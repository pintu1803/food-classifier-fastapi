from pydantic import BaseModel

#pair of label : confidence
class PredictionItem(BaseModel):
    label: str
    confidence: float

#list of such pairs.
class PredictionResponse(BaseModel):
    request_id: str
    prediction: str
    confidence: float
    latency: float
    top3: list[PredictionItem]