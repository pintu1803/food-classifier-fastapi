import os
from pathlib import Path
from dataclasses import dataclass

#Gobal code gets executed when we import this file.
#model and checkpoint dir are sibling of app dir

print("Current workig dir : ",os.getcwd())
BASE_DIR = Path(__file__).resolve().parent.parent
print("Project Base dir : ", BASE_DIR)


@dataclass
class PATH:
    CHECKPOINT_DIR: str = BASE_DIR / "checkpoints"
    DOWNLOAD_MODEL_PATH: str = BASE_DIR / "model"
    # DATASET_DIR: str = BASE_DIR / "dataset"
    CHECKPOINT_PATH_FOR_LOAD: str = CHECKPOINT_DIR / "best_val_acc_model_2.pth"
    
    
@dataclass
class IMAGE:
    height: int = 224
    width: int = 224
    classes: int = 15
