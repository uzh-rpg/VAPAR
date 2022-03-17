import os
import cv2
import numpy as np
import pandas as pd
import json

from tqdm import tqdm
from gazesim.data.utils import run_info_to_path


def compute_statistics(config):
    df_split = pd.read_csv(config["split"] + ".csv")
    df_index = pd.read_csv(os.path.join(config["data_root"], "index", "frame_index.csv"))
    df_index = df_index[df_split["split"] == "train"]

    # check if there is a faster way to compute the whole thing
    frame_stats_computed = False
    frame_stats_path = os.path.join(config["data_root"], "index", "frame_stats.csv")
    df_frame_stats = None
    if os.path.exists(frame_stats_path):
        df_frame_stats = pd.read_csv(frame_stats_path)
        if any([config["video_name"] in c for c in df_frame_stats.columns]):
            frame_stats_computed = True

    if frame_stats_computed:
        # do things the fast way
        df_frame_stats = df_frame_stats[df_split["split"] == "train"]

        # compute the mean
        data_mean = np.array([df_frame_stats["{}_channel_0".format(config["video_name"])].sum(),
                              df_frame_stats["{}_channel_0".format(config["video_name"])].sum(),
                              df_frame_stats["{}_channel_0".format(config["video_name"])].sum()])
        data_mean /= len(df_frame_stats.index)

        # compute the standard deviation
        data_sq_mean = np.array([df_frame_stats["{}_channel_0_sq".format(config["video_name"])].sum(),
                                 df_frame_stats["{}_channel_0_sq".format(config["video_name"])].sum(),
                                 df_frame_stats["{}_channel_0_sq".format(config["video_name"])].sum()])
        data_sq_mean /= len(df_frame_stats.index)
        data_std = np.sqrt(data_sq_mean - (data_mean ** 2))
    else:
        # do things the slow way
        cap_dict = {}

        # compute the mean first (required for std)
        data_mean = np.array([0.0, 0.0, 0.0])
        data_mean_sq = np.array([0.0, 0.0, 0.0])
        for i, row in tqdm(df_index.iterrows(), disable=False, total=len(df_index.index)):
            full_run_path = os.path.join(config["data_root"],
                                         run_info_to_path(row["subject"], row["run"], row["track_name"]))
            if full_run_path not in cap_dict:
                cap_dict[full_run_path] = cv2.VideoCapture(os.path.join(full_run_path,
                                                                        "{}.mp4".format(config["video_name"])))

            cap_dict[full_run_path].set(cv2.CAP_PROP_POS_FRAMES, row["frame"])

            data = cv2.cvtColor(cap_dict[full_run_path].read()[1], cv2.COLOR_BGR2RGB)
            data = data.astype("float64")
            if not config["original_range"]:
                data /= 255.0

            data_mean += np.mean(data, axis=(0, 1))
            data_mean_sq += np.mean(data ** 2, axis=(0, 1))
        data_mean /= len(df_index.index)
        data_mean_sq /= len(df_index.index)
        data_std = np.sqrt(data_mean_sq - (data_mean ** 2))

        """
        # compute the standard deviation, basically using the same loop
        data_std = None
        for i, row in tqdm(df_index.iterrows(), disable=False, total=len(df_index.index)):
            full_run_path = os.path.join(config["data_root"],
                                         run_info_to_path(row["subject"], row["run"], row["track_name"]))

            cap_dict[full_run_path].set(cv2.CAP_PROP_POS_FRAMES, row["frame"])

            data = cv2.cvtColor(cap_dict[full_run_path].read()[1], cv2.COLOR_BGR2RGB)
            data = data.astype("float64")
            if not config["original_range"]:
                data /= 255.0

            if data_std is None:
                data_std = np.mean((data - data_mean) ** 2, axis=(0, 1))
            else:
                data_std += np.mean((data - data_mean) ** 2, axis=(0, 1))
        data_std /= len(df_index.index)
        data_std = np.sqrt(data_std)
        """

    # save to the JSON file
    with open(config["split"] + "_info.json", "r") as f:
        split_info = json.load(f)
    if "mean" not in split_info:
        split_info["mean"] = {}
        split_info["std"] = {}
    split_info["mean"][config["video_name"]] = data_mean.tolist()
    split_info["std"][config["video_name"]] = data_std.tolist()
    with open(config["split"] + "_info.json", "w") as f:
        json.dump(split_info, f)


def parse_config(args):
    config = vars(args)
    try:
        split_index = int(config["split"])
        config["split"] = os.path.join(config["data_root"], "splits", "split{:03d}".format(split_index))
    except ValueError:
        if config["split"].endswith(".json"):
            config["split"] = os.path.abspath(config["split"])[:-5]
        elif config["split"].endswith(".csv"):
            config["split"] = os.path.abspath(config["split"])[:-4]
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-v", "--video_name", type=str, default="screen",
                        help="Video (masked or others).")
    parser.add_argument("-s", "--split", default=0,
                        help="The split of the data to compute statistics for (on the training set). "
                             "Can either be the path to a file or an index.")
    parser.add_argument("-or", "--original_range", action="store_true",
                        help="Whether to convert from [0, 255] to [0, 1] range before computing the statistics.")

    # parse the arguments
    arguments = parser.parse_args()

    # compute the statistics
    compute_statistics(parse_config(arguments))
