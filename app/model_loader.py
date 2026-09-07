from torchvision.models import resnet18
from torchvision import transforms
from utils import myLog
import sys, os, torch
from config import PATH, IMAGE

#=====================================================
torch.hub.set_dir(PATH.DOWNLOAD_MODEL_PATH)
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, IMAGE.classes)
model.eval()

best_model_load_path = PATH.CHECKPOINT_PATH_FOR_LOAD
if not os.path.exists(best_model_load_path):
    myLog("No checkpoint found. Can't predict.")
    sys.exit(1)

myLog("Checkpoint found. Using Model for Inference")
checkpoint = torch.load(best_model_load_path)
model.load_state_dict(checkpoint["model_state_dict"])
myLog("MODEL LOADED")
#=====================================================