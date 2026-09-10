from torchvision.models import resnet18
# from torchvision import transforms
from app.utils import myLog
import os
# import sys
import torch
from app.config import PATH, IMAGE
from gc import collect

from psutil import Process

model = None

#=====================================================
# from huggingface_hub import hf_hub_download
#=====================================================
#THIS IS NEEDED FOR RENDER DEPLOYMENT
#=====================================================
# best_model_load_path = hf_hub_download(
#     repo_id="pintusaini2979/resnet18-food-classifier",
#     filename="resnet18_food_deploy.pth"
# )
#=====================================================
def give_model_architecture():
    # torch.hub.set_dir(PATH.DOWNLOAD_MODEL_PATH)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, IMAGE.classes)
    model.eval()
    return model

#=====================================================
#THIS IS NEEDED FOR LOCAL INFERENCE
#=====================================================
def give_loaded_model():
    global model
    if model is not None:
        return model
    
    best_model_load_path = PATH.CHECKPOINT_PATH_FOR_LOAD

    if not os.path.exists(best_model_load_path):
        myLog(f"No checkpoint found in the {best_model_load_path}")
        return model
    else:
        model = give_model_architecture()
        myLog("Checkpoint found. Using Model for Inference")
        checkpoint = torch.load(best_model_load_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        myLog("MODEL LOADED")

        process = Process(os.getpid())
        myLog(
            f"RAM after model load: "
            f"{process.memory_info().rss / 1024 / 1024:.2f} MB"
        )
        #=====================================================
        #To save memory on render, delete checkpoint after dict loading
        del checkpoint
        collect()

        return model
#=====================================================

