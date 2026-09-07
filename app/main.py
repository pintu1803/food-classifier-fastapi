from fastapi import FastAPI, UploadFile, File, HTTPException
from inference import predict_image
from app.schema import PredictionResponse
from app.validators import validate_image, validate_size
from PIL import UnidentifiedImageError, Image
import io 
import uuid
from fastapi import Request


app = FastAPI(title="Indian Food Classifier API",
              version="1.0")

@app.get("/")
def home():
    return {
        "message" : "Indian Food Classifier API Running"
    }

@app.get("/health")
def health():
    return {
        "status" : "healthy"
    }

#add middleware to add request id to the request
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = uuid.uuid4()

    request.state.request_id = request_id

    response = await call_next(request)

    response.headers["X-request-id"] = request_id

    return response

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file : UploadFile = File(...)):
    #validate input type
    await validate_image(file)

    #convert bytes into image
    image_bytes = await file.read()

    #open with care
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )   

    #validate image size once type is confirmed
    await validate_size(image_bytes)

    #fetch request-id from the Request
    request_id = request.state.request_id

    #pass input to model for inference
    result = predict_image(image_bytes, request_id)

    #return response
    return result


