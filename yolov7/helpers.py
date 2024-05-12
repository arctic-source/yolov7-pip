# https://github.com/fcakyon/yolov5-pip/blob/main/yolov5/helpers.py

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from pathlib import Path

from PIL import Image

from yolov7.models.common import autoShape
from yolov7.models.experimental import attempt_load
from yolov7.utils.google_utils import attempt_download_from_hub, attempt_download
from yolov7.utils.torch_utils import TracedModel


def load_model(model_path_or_buffer, autoshape=True, device='cpu', trace=False, size=640, half=False, hf_model=False):
    # Adapted so model_file can be file_like object (weights in memory) or path to weights on disk
    if hf_model:
        model_file = attempt_download_from_hub(model_path_or_buffer)
    else:
        model_file = model_path_or_buffer  # Directly pass the file-like object or path

    model = attempt_load(model_file, map_location=device)
    if trace:
        model = TracedModel(model, device, size)

    if autoshape:
        model = autoShape(model)

    if half:
        model.half()

    return model



if __name__ == "__main__":
    model_path = "yolov7.pt"
    device = "cuda:0"
    model = load_model(model_path, device, trace=False, size=640, hf_model=False)
    imgs = [Image.open(x) for x in Path("inference/images").glob("*.jpg")]
    results = model(imgs, size=640, augment=False)
