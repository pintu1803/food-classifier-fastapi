from app.inference import predict_image, ModelNotLoadedError
from app.schema import PredictionResponse
from app.validators import validate_image, validate_size
from PIL import UnidentifiedImageError, Image
import io, os
from psutil import Process
import uuid
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException

import time 
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

@app.get("/memory")
def memory():
    process = Process(os.getpid())
    return {
        "ram_mb": process.memory_info().rss / 1024 / 1024
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

print("Origin of the frontend server : ", origins)
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

def latency(start, i):
    lat = round( (time.perf_counter() - start)*1000, 2)
    print("API Latency after step-",i," = ", lat)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file : UploadFile = File(...)):
    start = time.perf_counter()
    #validate input type
    await validate_image(file)
    latency(start, 1)

    #convert bytes into image
    image_bytes = await file.read()
    latency(start, 2)

    #open with care
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )   

    latency(start, 3)
    #validate image size once type is confirmed
    await validate_size(image_bytes)

    #fetch request-id from the Request
    request_id = request.state.request_id
    latency(start, 4)

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
            detail=str(e)
            # detail="Internal Server Error Occurred"
        )
    latency(start, 5)
    #return response
    return result


