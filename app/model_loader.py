from torchvision.models import resnet18
from torchvision import transforms
from app.utils import myLog
import sys, os, torch
from app.config import PATH, IMAGE

#=====================================================
from huggingface_hub import hf_hub_download
#=====================================================
#THIS IS NEEDED FOR RENDER DEPLOYMENT
#=====================================================
best_model_load_path = hf_hub_download(
    repo_id="pintusaini2979/resnet18-food-classifier",
    filename="resnet18_food_deploy.pth"
)
#=====================================================

# torch.hub.set_dir(PATH.DOWNLOAD_MODEL_PATH)
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, IMAGE.classes)
model.eval()

#=====================================================
#THIS IS NEEDED FOR LOCAL INFERENCE
#=====================================================
# best_model_load_path = PATH.CHECKPOINT_PATH_FOR_LOAD
# if not os.path.exists(best_model_load_path):
#     myLog("No checkpoint found. Can't predict.")
#     sys.exit(1)
#=====================================================

myLog("Checkpoint found. Using Model for Inference")
checkpoint = torch.load(best_model_load_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
myLog("MODEL LOADED")
#=====================================================
#To save memory on render, delete checkpoint after dict loading
del checkpoint
import gc
gc.collect()

