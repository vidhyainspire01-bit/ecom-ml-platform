import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_scoring_dir = os.path.dirname(_this_dir)
for p in (_this_dir, _scoring_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.model_loader import load_model
from common.inference_runner import run as _run
from preprocessing import preprocess

model = None


def init():
    global model
    model = load_model()


def run(raw_data):
    return _run(raw_data, model, preprocess)