import os
from pathlib import Path
from dataclasses import dataclass

#Gobal code gets executed when we import this file.
#model and checkpoint dir are sibling of app dir

print("Current working dir : ",os.getcwd())
BASE_DIR = Path(__file__).resolve().parent.parent
print("Project Base dir : ", BASE_DIR)


@dataclass
class PATH:
    CHECKPOINT_DIR: Path = BASE_DIR / "checkpoints"
    DOWNLOAD_MODEL_PATH: Path = BASE_DIR / "model"
    # DATASET_DIR: Path = BASE_DIR / "dataset"
    CHECKPOINT_PATH_FOR_LOAD: Path = CHECKPOINT_DIR / "best_val_acc_model_2.pth"
    CHECKPOINT_PATH_FOR_SAVE: Path = CHECKPOINT_DIR / "resnet18_food_deploy.pth"
    
    
@dataclass
class IMAGE:
    height: int = 224
    width: int = 224
    classes: int = 15
