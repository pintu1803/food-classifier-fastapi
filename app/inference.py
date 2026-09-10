from app.model_loader import give_loaded_model
import torch
from PIL import Image
import time
from app.utils import preprocess, get_food_class

"""
1. Load the model only once on startup and store it in cache and use for all the inferences. 
2. predict_image gets called by FastAPI and receive byte object
3. converts byte object into PIL and preprocess it.
4. Pass it to model and get prediction.
5. Find softmax and get top K probabilities
6. Construct a dict with label and confidence
7. return the python dict, fast api will handle it
"""

model = give_loaded_model()

class ModelNotLoadedError(Exception):
    pass

def ensure_model_load():
    global model
    if model is None:
        raise ModelNotLoadedError("Checkpoint not found.")


def predict_image(pil_image, request_id):
    #ensure model is correctly loaded
    ensure_model_load()

    #1. measure latency
    start = time.perf_counter()

    #2. preprocess the image
    image = preprocess(pil_image)

    #3. add the extra dim for making a batch of size 1
    image = torch.unsqueeze(input=image, dim=0)

    #4. feed input to model and get prediction
    with torch.inference_mode():
        pred = model(image)

    #measure latency
    latency = (time.perf_counter() - start)*1000

    #5. Squeeze the extra dim (batch dim) from pred.
    pred = torch.squeeze(input=pred, dim=0)

    #6. convert raw logits into probabilities
    prob = torch.softmax(input=pred, dim=0)

    #7.Fetch top K prob
    topk_probs, topk_index = torch.topk(prob, k=3)

    #8. convert tensors to list
    topk_probs = topk_probs.tolist()
    topk_index = topk_index.tolist()

    #9. Fetch top K food class names
    topk_names = get_food_class(topk_index)

    #10. make top K list 
    topK_list = []
    for i in range(len(topk_probs)):
        topK_list.append({"label": topk_names[i], "confidence": topk_probs[i]})

    #Construct full response
    response = {
        "request_id": request_id,
        "prediction": topk_names[0],
        "confidence": topk_probs[0],
        "latency": round(latency, 2),
        "topK": topK_list
    }


    #11. return the dict, FASTAPI will handle it
    return response
#=====================================================
