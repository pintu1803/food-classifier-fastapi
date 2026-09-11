from app.utils import myLog
import os
import onnxruntime as ort
from app.config import PATH, IMAGE
from gc import collect
from psutil import Process
session = None

#=====================================================
#THIS IS NEEDED FOR LOCAL INFERENCE
#=====================================================
def give_loaded_model():
    global session
    if session is not None:
        return session
    
    best_model_load_path = PATH.ONNX_MODEL_PATH

    if not os.path.exists(best_model_load_path):
        # myLog(f"No checkpoint found in the {best_model_load_path}")
        return session
    else:
        sess_options = ort.SessionOptions()
        # Cap threads explicitly - onnxruntime has the same thread-oversubscription
        # risk on a CPU-throttled container that plain torch did.
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
    
        session = ort.InferenceSession(
            str(best_model_load_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        myLog("MODEL LOADED")

        process = Process(os.getpid())
        myLog(
            f"RAM after session load: "
            f"{process.memory_info().rss / 1024 / 1024:.2f} MB"
        )

        return session
#=====================================================

# import torch
# from torchvision.models import resnet18

# model = resnet18(weights=None)
# model.fc = torch.nn.Linear(model.fc.in_features, IMAGE.classes)

# best_model_load_path = PATH.CHECKPOINT_PATH_FOR_LOAD
# checkpoint = torch.load(best_model_load_path, map_location="cpu")
# model.load_state_dict(checkpoint)

# ONNX_PATH = "models/food_classifier.onnx"

# dummy_input = torch.randn(1, 3, IMAGE.height, IMAGE.width)

# torch.onnx.export(
#         model,
#         dummy_input,
#         str(PATH.ONNX_MODEL_PATH),
#         input_names=["input"],
#         output_names=["output"],
#         opset_version=17,
#         dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
#         do_constant_folding=True,
#     )