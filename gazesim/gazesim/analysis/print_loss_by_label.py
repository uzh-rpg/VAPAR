import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from pprint import pprint
from gazesim.data.utils import resolve_split_index_path
from gazesim.training.config import parse_config as parse_train_config
from gazesim.training.helpers import resolve_model_class, resolve_dataset_class
from gazesim.training.utils import to_device


def generate(config):
    # load frame_index and split_index
    frame_index = pd.read_csv(os.path.join(config["data_root"], "index", "frame_index.csv"))
    split_index = pd.read_csv(config["split_config"] + ".csv")

    # use GPU if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(config["gpu"])
                          if use_cuda and config["gpu"] < torch.cuda.device_count() else "cpu")

    # define paths
    log_dir = os.path.abspath(os.path.join(os.path.dirname(config["model_load_path"]), os.pardir))
    save_dir = os.path.join(log_dir, "predictions")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config_path = os.path.join(log_dir, "config.json")

    # load the config
    with open(config_path, "r") as f:
        train_config = json.load(f)
    train_config["data_root"] = config["data_root"]
    train_config["gpu"] = config["gpu"]
    train_config["model_load_path"] = config["model_load_path"]
    train_config = parse_train_config(train_config)
    train_config["split_config"] = config["split_config"]

    # load the model
    model_info = train_config["model_info"]
    model = resolve_model_class(train_config["model_name"])(train_config)
    model.load_state_dict(model_info["model_state_dict"])
    model = model.to(device)
    model.eval()

    # losses
    loss_function_mse = torch.nn.MSELoss(reduction="none")
    loss_function_l1 = torch.nn.L1Loss(reduction="none")

    # some info for saving the predictions
    experiment_name = train_config["experiment_name"][20:]
    epoch = model_info["epoch"]

    label_dict = {
        0: "flat_left",
        1: "flat_right",
        2: "wave_left",
        3: "wave_right",
        4: "none"
    }

    for split in config["split"]:
        current_dataset = resolve_dataset_class(train_config["dataset_name"])(train_config, split=split)
        current_dataloader = DataLoader(current_dataset, batch_size=config["batch_size"],
                                        shuffle=False, num_workers=config["num_workers"])

        loss_accumulator = {v: {"mse": [], "l1": []} for k, v in label_dict.items()}

        for batch_index, batch in tqdm(enumerate(current_dataloader), total=len(current_dataloader)):
            # transfer to GPU
            batch = to_device(batch, device)

            # forward pass and loss computation
            predictions = model(batch)
            # TODO: for now, just compute MSE manually and for each element separately
            loss_mse = loss_function_mse(predictions["output_control"], batch["output_control"])
            loss_l1 = loss_function_l1(predictions["output_control"], batch["output_control"])

            loss_mse = loss_mse.mean(dim=-1).cpu().detach().numpy()
            loss_l1 = loss_l1.mean(dim=-1).cpu().detach().numpy()

            # accumulate losses
            for l_idx, label in enumerate(batch["label_high_level"].cpu().detach().numpy()):
                loss_accumulator[label_dict[label]]["mse"].append(loss_mse[l_idx])
                loss_accumulator[label_dict[label]]["l1"].append(loss_l1[l_idx])

        for k, v in loss_accumulator.items():
            if len(v["mse"]) > 0:
                v["mse"] = np.mean(v["mse"])
            if len(v["l1"]) > 0:
                v["l1"] = np.mean(v["l1"])

        pprint(loss_accumulator)


def parse_config(args):
    config = vars(args)
    config["split_config"] = resolve_split_index_path(config["split_config"], config["data_root"])
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-m", "--model_load_path", type=str, default=None,
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-s", "--split", type=str, nargs="+", default=["val"], choices=["train", "val", "test"],
                        help="Splits for which to create videos.")
    parser.add_argument("-sc", "--split_config", type=str, default=0,
                        help="TODO.")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="The GPU to use.")
    parser.add_argument("-w", "--num_workers", type=int, default=4,
                        help="Number of workers to use for loading the data.")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="Batch size to use for training.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    generate(parse_config(arguments))

