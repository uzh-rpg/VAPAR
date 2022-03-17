import json
import os

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from gazesim.training.config import parse_config
from gazesim.training.helpers import resolve_model_class


def get_batch_size(batch):
    if isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
        return batch.shape[0]
    elif isinstance(batch, dict):
        return get_batch_size(batch[list(batch.keys())[0]])
    elif isinstance(batch, list):
        return get_batch_size(batch[0])
    return np.nan


def to_batch(batch):
    return default_collate(batch)


def to_device(batch, device, make_batch=False):
    # TODO: maybe add something for a list too...
    for k in batch:
        if torch.is_tensor(batch[k]):
            if make_batch:
                batch[k] = batch[k].unsqueeze(0)
            batch[k] = batch[k].to(device)
        elif isinstance(batch[k], dict):
            batch[k] = to_device(batch[k], device, make_batch)
    return batch


def load_model(checkpoint_path, gpu=-1, return_config=False):
    # define paths
    log_dir = os.path.abspath(os.path.join(os.path.dirname(checkpoint_path), os.pardir))
    config_path = os.path.join(log_dir, "config.json")

    # load the config (only needed to get info, probably not returned with the model?)
    with open(config_path, "r") as f:
        train_config = json.load(f)
    train_config["gpu"] = gpu
    train_config["model_load_path"] = checkpoint_path
    train_config = parse_config(train_config)

    # load the model
    model_info = train_config["model_info"]
    model = resolve_model_class(train_config["model_name"])(train_config)
    model.load_state_dict(model_info["model_state_dict"])
    # model.load_model_info(model_info)
    # model = model.to(device)  # I guess device should probably be identified outside of the thingy?

    if return_config:
        train_config["model_info"] = None  # just to save memory, might be removed if necessary
        return model, train_config
    return model
