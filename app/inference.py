from model_loader import model
import torch
from PIL import Image
import io
from utils import preprocess, get_food_class

"""
1. Load the model only once on startup and store it in cache and use for all the inferences. 
2. predict_image gets called by FastAPI and receive byte object
3. converts byte object into PIL and preprocess it.
4. Pass it to model and get prediction.
5. Find softmax and get top K probabilities
6. Construct a dict with label and confidence
7. return the python dict, fast api will handle it
"""

def predict_image(input_bytes):
    #1. Convert the bytes in PIL
    pil_image = Image.open(io.BytesIO(input_bytes))

    #2. preprocess the image
    image = preprocess(pil_image)

    #3. add the extra dim for making a batch of size 1
    image = torch.unsqueeze(input=image, dim=0)

    #4. feed input to model and get prediction
    with torch.inference_mode():
        pred = model(image)

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

    #10. make dict {"label": "confidence"}
    result_dict = {topk_names[i] : topk_probs[i] for i in range(len(topk_probs))}

    #11. return the dict, FASTAPI will handle it
    return result_dict
#=====================================================
