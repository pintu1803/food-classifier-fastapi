from torchvision import transforms
from app.config import IMAGE
import numpy as np
from PIL import Image

def myLog(title):
    margin = "*" * ((78 - len(title))//2)
    print("\n")
    print("=" * 80)
    print(f"{margin} {title} {margin}")
    print("=" * 80)

def preprocess(image):
    """We don't transform the validation/testing data from augmentation pov, 
    only training data exclusively gets transformed.
    However, we need to resize and convert the testing data as per resnet architecture."""
    IMAGENET_MEAN=[0.485, 0.456, 0.406],
    IMAGENET_STD=[0.229, 0.224, 0.225]

    image = image.convert("RGB")
    image = image.resize((IMAGE.width, IMAGE.height), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return np.ascontiguousarray(arr, dtype=np.float32)

def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
 
def topk(probs: np.ndarray, k: int):
    #get list of original indices, array sorted in descending order 
    indices = np.argsort(probs)[::-1]
    top_indices = indices[:k]
    top_values = probs[top_indices]
    return top_values, top_indices

def get_food_class(indices):
    classes =  ['biryani', 'cholebhature', 'dabeli', 'dal', 'dhokla', 
                'dosa', 'jalebi', 'kathiroll', 'kofta', 'naan', 
                'pakora', 'paneer', 'panipuri', 'pavbhaji', 'vadapav']

    return [classes[index] for index in indices]