"""
Contains all of the necessary utilities and functions to run prediction for the gradio app.
"""

from typing import Any, Union, Dict, Tuple

import torch
import librosa
import gradio as gr

from beats import BEATs, BEATsConfig
from gr_config import config, BaseConfig

# CPU/GPU Configurations
if torch.cuda.is_available():
    DEVICE = [0]  # use 0th CUDA device
    ACCELERATOR = "gpu"
else:
    DEVICE = 1
    ACCELERATOR = "cpu"

MAP_LOCATION: str = torch.device(f"cuda:{DEVICE[0]}" if ACCELERATOR == "gpu" else "cpu")

# Gradio Input/Output Configurations
inputs: Union[str, gr.inputs.Audio] = gr.inputs.Audio(source="upload", type="filepath")
outputs: Union[str, gr.outputs.Label] = "label"


# Helper functions
def initialize_esc_model(cfg: BaseConfig) -> Tuple[BEATs, Dict[str, Any]]:
    """
    loads and initialises the esc model
    """

    # load the fine-tuned checkpoints
    checkpoint = torch.load(cfg.esc_model_path)

    cfg = BEATsConfig(checkpoint["cfg"])
    beats_model = BEATs(cfg)
    beats_model.load_state_dict(checkpoint["model"])
    beats_model = beats_model.eval()

    return beats_model, checkpoint["label_dict"]


def load_label_mapping(cfg: BaseConfig) -> Dict[str, str]:
    """
    loads the mapping from the labels file
    """

    with open(cfg.labels_path, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
    items = [line.split(",") for line in lines[1:]]

    return {x[1]: x[2].strip("\r\n").replace('"', "") for x in items}


# Initialize models
esc_model, esc_label_dict = initialize_esc_model(config)
mapping = load_label_mapping(config)


# Main prediction function
def predict(audio_path: str) -> str:
    """
    takes in an audio file path and outputs the top-k labels
    """

    arr, _ = librosa.load(audio_path, sr=16000, mono=True)
    torch_arr = torch.from_numpy(arr)
    torch_arr = torch_arr.unsqueeze(0)
    padding_mask = torch.zeros(torch_arr.shape).bool()

    with torch.no_grad():
        probs = esc_model.extract_features(torch_arr, padding_mask=padding_mask)[0]

    topk_label_prob, topk_label_idx = probs.topk(k=config.topk)
    topk_label = [
        mapping[esc_label_dict[label_idx.item()]] for label_idx in topk_label_idx[0]
    ]
    topk_prob = topk_label_prob[0]

    return {label: prob.item() for label, prob in zip(topk_label, topk_prob)}
