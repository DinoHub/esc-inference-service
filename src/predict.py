import os
import logging
from typing import Any, List, Optional, Union, Dict, Tuple

import torch
import librosa
import gradio as gr
import soundfile as sf

from beats import BEATs, BEATsConfig
from config import config, BaseConfig

''' CPU/GPU Configurations '''
if torch.cuda.is_available():
    DEVICE = [0]  # use 0th CUDA device
    ACCELERATOR = 'gpu'
else:
    DEVICE = 1
    ACCELERATOR = 'cpu'

MAP_LOCATION: str = torch.device('cuda:{}'.format(DEVICE[0]) if ACCELERATOR == 'gpu' else 'cpu')

''' Gradio Input/Output Configurations '''
inputs: Union[str, gr.Audio] = gr.Audio(source='upload', type='filepath')
outputs: gr.Label = gr.Label(show_label=False)

''' Helper functions '''
def initialize_esc_model(cfg: BaseConfig) -> Tuple[BEATs, Dict[str, Any]]:

    # load the fine-tuned checkpoints
    checkpoint = torch.load(cfg.esc_model_path)

    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model = BEATs_model.eval()

    return BEATs_model, checkpoint['label_dict']

def load_label_mapping(cfg: BaseConfig) -> Dict[str, str]:

    with open(cfg.labels_path, mode='r') as f:
        lines = f.readlines()
    items = [line.split(',') for line in lines[1:]]

    return {x[1]:x[2].strip('\r\n').replace('"','') for x in items}

''' Initialize models '''
esc_model, esc_label_dict = initialize_esc_model(config)
mapping = load_label_mapping(config)

''' Main prediction function '''
def predict(audio_path: str) -> str:

    arr, sr = librosa.load(audio_path, sr=16000, mono=True)
    torch_arr = torch.from_numpy(arr)
    torch_arr = torch_arr.unsqueeze(0)
    padding_mask = torch.zeros(torch_arr.shape).bool()

    with torch.no_grad():
        probs = esc_model.extract_features(torch_arr, padding_mask=padding_mask)[0]

    top8_label_prob, top8_label_idx = probs.topk(k=8)
    top8_label = [mapping[esc_label_dict[label_idx.item()]] for label_idx in top8_label_idx[0]]
    top8_prob = top8_label_prob[0]
    return {label:prob.item() for label, prob in zip(top8_label, top8_prob)}
