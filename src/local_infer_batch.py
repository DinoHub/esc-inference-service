"""
Code to run inference on a batch of audio files. Require a JSON manifest file in this format:

manifest.json
================
{"audio_filepath": "rel/path/to/audio1.wav"}
{"audio_filepath": "rel/path/to/audio2.wav"}
{"audio_filepath": "rel/path/to/audio3.wav"}

"""

import os
import json
from typing import Any, Dict, Tuple, List

import torch
import librosa
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from beats import BEATs, BEATsConfig
from local_config import config, BaseConfig

# CPU/GPU Configurations
if torch.cuda.is_available():
    DEVICE = [0]  # use 0th CUDA device
    ACCELERATOR = "gpu"
else:
    DEVICE = 1
    ACCELERATOR = "cpu"

MAP_LOCATION: str = torch.device(f"cuda:{DEVICE[0]}" if ACCELERATOR == "gpu" else "cpu")


class CustomAudioDataset(Dataset):
    """
    Custom audio dataset that takes in a list of manifest items as input.
    List should look like this:
    [{"audio_filepath": "rel/path/to/audio.wav"}, ...]
    """

    def __init__(self, data_dir: str, manifest_list: List[Dict[str, str]]):
        """
        Arguments:
            data_dir (str): data directory where the manifest file lies
            manifest_list (List[Dict[str, str]]): List of entries (manifest should be loaded prior)
        """
        self.data_dir = data_dir
        self.data = manifest_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # loads and convert audio array into tensor
        audio_filepath = os.path.join(self.data_dir, self.data[idx]["audio_filepath"])
        arr, _ = librosa.load(audio_filepath, sr=16000, mono=True)
        torch_arr = torch.from_numpy(arr)

        return torch_arr


def custom_collate_fn(tensor_batch: List[torch.Tensor]):
    """
    within a batch, pad sequences to the batch's max length
    """
    return pad_sequence(tensor_batch, batch_first=True)


def get_mappers(cfg: BaseConfig) -> Tuple[Dict[str, str]]:
    """
    loads the mapping from the labels file

    output:
    (
        {/m/07pyy8b: "Pant", ...}, # label2name
        {"Pant": /m/07pyy8b, ...}, # name2label
    )

    """

    with open(cfg.labels_path, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
    items = [line.split(",") for line in lines[1:]]

    return (
        {x[1]: x[2].strip("\r\n").replace('"', "") for x in items},
        {x[2].strip("\r\n").replace('"', ""): x[1] for x in items},
    )


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


if __name__ == "__main__":
    # load the audio data into a dataset
    with open(config.manifest_path, mode="r", encoding="utf-8") as fr:
        entries = [json.loads(entry) for entry in fr.readlines()]
    manifest_dir = os.path.dirname(config.manifest_path)
    dataset = CustomAudioDataset(manifest_dir, entries)

    # set up the data loader with the custom collate fn
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    # initialise esc model
    esc_model, esc_label_dict = initialize_esc_model(config)
    esc_model.to(MAP_LOCATION)
    esc_label_dict_reverse = {v: k for k, v in esc_label_dict.items()}

    # mappers
    label2name, name2label = get_mappers(config)
    target_labels = [name2label[name] for name in config.target_classes]
    target_indices = [esc_label_dict_reverse[label] for label in target_labels]

    outputs = []

    for batch in tqdm(loader):
        batch = batch.to(MAP_LOCATION)
        padding_mask = torch.zeros(batch.shape).bool().to(MAP_LOCATION)
        with torch.no_grad():
            probs = esc_model.extract_features(batch, padding_mask=padding_mask)[0]

        if config.mode == "target":
            # only output probabilities of target classes in output manifest
            for i, prob in enumerate(probs):
                output = {}
                for target_name, target_idx in zip(
                    config.target_classes, target_indices
                ):
                    output[target_name] = prob[target_indices].item()

                outputs.append(output)

        elif config.mode == "topk":
            # output probabilities of top-k classes in output manifest
            for i, (topk_label_prob, topk_label_idx) in enumerate(
                zip(*probs.topk(k=config.topk))
            ):
                topk_label = [
                    label2name[esc_label_dict[label_idx.item()]]
                    for label_idx in topk_label_idx[0]
                ]
                topk_prob = topk_label_prob[0]

                outputs.append(
                    {label: prob.item() for label, prob in zip(topk_label, topk_prob)}
                )

    assert len(entries) == len(outputs), "Outputs and entries length mismatch???"

    with open(config.output_path, mode="w", encoding="utf-8") as fw:
        for entry, output in zip(entries, outputs):
            entry["output"] = output
            fw.write(json.dumps(entry)+"\n")
