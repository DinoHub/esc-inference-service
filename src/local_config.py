"""
This file contains all of the configurations to run the inference locally.
"""

from typing import List
from pydantic import BaseSettings


class BaseConfig(BaseSettings):
    """Define any config here.

    See here for documentation:
    https://pydantic-docs.helpmanual.io/usage/settings/
    """

    # data manifest path
    manifest_path: str = "/dataset/manifest.json"
    output_path: str = "/workspace/output.json"
    batch_size: int = 8

    esc_model_path: str = "models/AS2M_beats.pt"
    labels_path: str = "misc/class_labels_indices.csv"

    mode: str = "target"  # either "target" or "topk"

    # please refer to misc/class_labels_indices.csv
    # for the classes the model was trained on
    target_classes: List[str] = [
        "Speech",
    ]
    topk: int = 10


config = BaseConfig()
