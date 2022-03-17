import os
import json
import pandas as pd
import torch

from torch.utils.data._utils.collate import default_collate as to_batch
from tqdm import tqdm
from gazesim.data.utils import find_contiguous_sequences, resolve_split_index_path, run_info_to_path
from gazesim.training.config import parse_config as parse_train_config
from gazesim.training.helpers import resolve_model_class, resolve_dataset_class
from gazesim.training.utils import to_device
from gazesim.training.helpers import resolve_output_processing_func


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

    # some info for saving the predictions
    experiment_name = train_config["experiment_name"][20:]
    epoch = model_info["epoch"]

    for split in config["split"]:
        current_frame_index = frame_index.loc[split_index["split"] == split]
        current_dataset = resolve_dataset_class(train_config["dataset_name"])(train_config, split=split)
        sequences = find_contiguous_sequences(current_frame_index, new_index=True)

        # test = list(current_frame_index.groupby(["track_name", "subject", "run"]).count().index)
        # could loop through the above, get stuff separately, get indices from current_frame_index...
        
        columns_gt = [c + "_gt" for c in current_dataset.output_columns]
        columns_pred = [c + "_pred" for c in current_dataset.output_columns]

        prediction_dict = {}
        # TODO: maybe use groupby instead of contiguous sequences and then take the index of that...
        run_dirs = [run_info_to_path(current_frame_index["subject"].iloc[si],
                                     current_frame_index["run"].iloc[si],
                                     current_frame_index["track_name"].iloc[si])
                    for si, _ in sequences]
        if len(config["subjects"]) > 0:
            check = [current_frame_index["subject"].iloc[si] in config["subjects"] for si, _ in sequences]
            sequences = [sequences[i] for i in range(len(sequences)) if check[i]]
            run_dirs = [run_dirs[i] for i in range(len(run_dirs)) if check[i]]
        if len(config["runs"]) > 0:
            check = [current_frame_index["run"].iloc[si] in config["runs"] for si, _ in sequences]
            sequences = [sequences[i] for i in range(len(sequences)) if check[i]]
            run_dirs = [run_dirs[i] for i in range(len(run_dirs)) if check[i]]

        # TODO: will probably have to change this, so that not only left/right curve are included...?
        """
        for track_name, subject, run in tqdm(test):
            hmm = current_frame_index[(current_frame_index["subject"] == subject) & (current_frame_index["run"] == run)]

            # get the directory
            rel_run_dir = run_info_to_path(subject, run, track_name)
            run_dir = os.path.join(save_dir, rel_run_dir)

            # load the screen timestamp file
            df_screen = pd.read_csv(os.path.join(config["data_root"], rel_run_dir, "screen_timestamps.csv"))
            df_pred = df_screen[["ts", "frame"]].copy()
            # df_pred = df_pred.set_index("frame")

            # loop through all frames?? and find their index in the current_frame_index?
            # maybe better to reset its index, further slice current_frame_index and then figure out which indices
            # we need from the dataframe?
            for i in range(len(df_pred.index)):
                print(hmm["frame"])
                ds_index = hmm.index[hmm["frame"] == df_pred["frame"].iloc[i]]
                print(ds_index)
                exit()
        """

        for (start_index, end_index), run_dir in tqdm(zip(sequences, run_dirs), disable=False, total=len(sequences)):
            if run_dir not in prediction_dict:
                # load screen ts dataframe
                df_screen = pd.read_csv(os.path.join(config["data_root"], run_dir, "screen_timestamps.csv"))
                prediction_dict[run_dir] = df_screen[["ts", "frame"]].copy()
                prediction_dict[run_dir] = prediction_dict[run_dir].set_index("frame")

            for index in tqdm(range(start_index, end_index), disable=True):
                frame = current_frame_index["frame"].iloc[index]

                # read the current data sample
                sample = to_batch([current_dataset[index]])
                sample = to_device(sample, device)

                # get the predictions
                prediction = model(sample)
                prediction["output_control"] = resolve_output_processing_func("output_control")(prediction["output_control"])

                # get the values as numpy arrays
                control_gt = sample["output_control"].cpu().detach().numpy().reshape(-1)
                control_prediction = prediction["output_control"].cpu().detach().numpy().reshape(-1)

                # store data in the dataframe
                prediction_dict[run_dir].loc[frame, columns_gt] = control_gt
                prediction_dict[run_dir].loc[frame, columns_pred] = control_prediction

        # save the dataframe to csv
        for rd, df in prediction_dict.items():
            cd = os.path.join(save_dir, rd)
            if not os.path.exists(cd):
                os.makedirs(cd)
            df.to_csv(os.path.join(cd, "predictions_{}_e{:03d}.csv".format(experiment_name, epoch)), index=False)


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
    parser.add_argument("-sub", "--subjects", type=int, nargs="*", default=[],
                        help="Subjects to use.")
    parser.add_argument("-run", "--runs", type=int, nargs="*", default=[],
                        help="Runs to use.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    generate(parse_config(arguments))

