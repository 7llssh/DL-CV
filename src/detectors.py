from __future__ import annotations
from ultralytics import YOLO

def load_model(weights_path: str, device: str = "cpu"):
    # ultralytics takes device in predict/train; model itself loads weights here
    return YOLO(weights_path)

def get_name_map(model) -> dict[int, str]:
    # model.names can be dict or list
    names = model.names
    if isinstance(names, dict):
        return names
    return {i: n for i, n in enumerate(names)}
