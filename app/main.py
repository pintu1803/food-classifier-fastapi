from fastapi import FastAPI, UploadFile, File
from inference import predict_image

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

@app.get("/predict")
async def predict(file : UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    return result


