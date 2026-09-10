from app.inference import predict_image, ModelNotLoadedError
from app.schema import PredictionResponse
from app.validators import validate_image, validate_size
from PIL import UnidentifiedImageError, Image
import io, os
import uuid
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException


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

#=====================================
ENV = os.getenv("ENV", "development")

if ENV == "production":
    origins = [
        "https://foodplateai.vercel.app"
    ]
else:
    origins = [
        "http://localhost:5173"
    ]

print("Origin of the backend server : ", origins)
#======================================

#add cors middleware to connect frontend with backend
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["*"],
                   allow_headers=["*"],)

#add middleware to add request id to the request
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())

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
    #use thread pool
    #manages concurrent requests
    #wrap it inside try-catch to handle 500 error
    try:
        result = await run_in_threadpool(
            predict_image, 
            pil_image, 
            request_id
            )
    except ModelNotLoadedError as e:
        raise HTTPException(
            status_code=503,
            detail="Model is not available on this deployment"
        )
    except Exception as e:
        raise HTTPException (
            status_code=500,
            detail="Internal Server Error Occurred"
        )

    #return response
    return result


