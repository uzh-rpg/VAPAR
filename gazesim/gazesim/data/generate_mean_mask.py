import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import multivariate_normal as mvn
from gazesim.data.utils import filter_by_screen_ts, filter_by_property, run_info_to_path, pair, generate_gaussian_heatmap

# TODO: probably change this to have one function that just takes in all the data it should use to generate
#  the mean mask (in the format of the global index file) => can then either load a split index file and take the
#  training samples or "recreate" the process of dynamically filtering (e.g. expected trajectory) and then splitting
#  the data, as I would prefer to do it when creating datasets (since that should create a split closer to the
#  desired input parameters)


def generate(args):
    # load the indexing dataframe from the root directory
    df_index = pd.read_csv(args.index_file)

    # define the dimensions of the output map and creating a scaled down heatmap
    sigma = np.array([[200, 0], [0, 200]])
    width = 800
    height = 600
    down_scale_factor = 5

    width_small = int(np.round(width / down_scale_factor))
    height_small = int(np.round(height / down_scale_factor))

    grid = np.mgrid[0:width_small, 0:height_small]
    grid = grid.transpose((1, 2, 0))

    # create the array to accumulate values in
    values_accumulated = np.zeros((width_small, height_small))

    # loop through the frames
    df_dict = {}
    for _, row in tqdm(df_index.iterrows(), total=len(df_index.index)):
        # load the dataframes if they haven't been loaded yet
        if row["rel_run_path"] not in df_dict:
            df_screen = pd.read_csv(os.path.join(args.data_root, row["rel_run_path"], "screen_timestamps.csv"))
            df_gaze = pd.read_csv(os.path.join(args.data_root, row["rel_run_path"], "gaze_on_surface.csv"))

            # filter out the frames that cannot be found in the index
            indexed_frames = df_index[df_index["rel_run_path"] == row["rel_run_path"]]["frame"]
            df_screen = df_screen[df_screen["frame"].isin(indexed_frames)]

            # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
            df_gaze = df_gaze[["ts", "frame", "norm_x_su", "norm_y_su"]]
            df_gaze.columns = ["ts", "frame", "x", "y"]
            df_gaze["x"] = df_gaze["x"] * 800
            df_gaze["y"] = (1.0 - df_gaze["y"]) * 600

            # filter by screen timestamps
            df_screen, df_gaze = filter_by_screen_ts(df_screen, df_gaze)

            df_dict[row["rel_run_path"]] = {"screen": df_screen, "gaze": df_gaze}

        df_gaze = df_dict[row["rel_run_path"]]["gaze"]
        current_gaze = df_gaze[df_gaze["frame"] == row["frame"]]
        for _, gaze_row in current_gaze.iterrows():
            mu = np.array([gaze_row["x"], gaze_row["y"]])
            gaussian = mvn(mean=(mu / down_scale_factor), cov=(sigma / (down_scale_factor ** 2)))
            values_current = gaussian.pdf(grid)
            values_accumulated += values_current

    values_accumulated = cv2.resize(values_accumulated.copy(), (height, width), interpolation=cv2.INTER_CUBIC)
    values_accumulated /= len(df_index.index)
    mean_mask = values_accumulated.transpose((1, 0))

    if mean_mask.max() > 0.0:
        mean_mask /= mean_mask.max()
    mean_mask = (mean_mask * 255).astype("uint8")
    mean_mask = np.repeat(mean_mask[:, :, np.newaxis], 3, axis=2)
    cv2.imwrite(os.path.join(args.data_root, f"mean_mask_{os.path.basename(args.index_file)[:-4]}.png"), mean_mask)


def get_mean_mask(data, config):
    # data should already be filtered etc.

    # define the dimensions of the output map and creating a scaled down heatmap
    # TODO: these should probably also be parameters that can be specified...
    sigma = np.array([[200, 0], [0, 200]])
    width = 800
    height = 600
    down_scale_factor = config["down_scale_factor"]

    width_small = int(np.round(width / down_scale_factor))
    height_small = int(np.round(height / down_scale_factor))

    grid = np.mgrid[0:width_small, 0:height_small]
    grid = grid.transpose((1, 2, 0))

    # create the array to accumulate values in
    weights = np.zeros((width_small, height_small))
    total = 0

    # loop through the frames
    df_dict = {}
    counter = 0
    for _, row in tqdm(data.iterrows(), total=len(data.index)):
        # load the dataframes if they haven't been loaded yet
        rel_run_path = run_info_to_path(row["subject"], row["run"], row["track_name"])
        if rel_run_path not in df_dict:
            df_screen = pd.read_csv(os.path.join(config["data_root"], rel_run_path, "screen_timestamps.csv"))
            df_gaze = pd.read_csv(os.path.join(config["data_root"], rel_run_path, "gaze_on_surface.csv"))

            # filter out the frames that cannot be found in the index
            indexed_frames = filter_by_property(data, [], {"subject": row["subject"],
                                                           "run": row["run"],
                                                           "track_name": row["track_name"]})["frame"]
            df_screen = df_screen[df_screen["frame"].isin(indexed_frames)]

            # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
            df_gaze = df_gaze[["ts", "frame", "norm_x_su", "norm_y_su"]]
            df_gaze.columns = ["ts", "frame", "x", "y"]
            df_gaze["x"] = np.round(df_gaze["x"] * 800).astype(int)
            df_gaze["y"] = np.round((1.0 - df_gaze["y"]) * 600).astype(int)

            # filter by screen timestamps
            df_screen, df_gaze = filter_by_screen_ts(df_screen, df_gaze)

            df_dict[rel_run_path] = {"screen": df_screen, "gaze": df_gaze}

        df_gaze = df_dict[rel_run_path]["gaze"]
        current_gaze = df_gaze[df_gaze["frame"] == row["frame"]]
        for _, gaze_row in current_gaze.iterrows():
            x = int(gaze_row["x"] / down_scale_factor)
            y = int(gaze_row["y"] / down_scale_factor)
            if 0 <= x < weights.shape[0] and 0 <= y < weights.shape[1]:
                weights[x, y] += 1
                total += 1
            # mu = np.array([gaze_row["x"], gaze_row["y"]])
            # gaussian = mvn(mean=(mu / down_scale_factor), cov=(sigma / (down_scale_factor ** 2)))
            # values_current = gaussian.pdf(grid)
            # values_accumulated += values_current
        counter += 1
        # if counter == 1:
        #     break

    # weights = np.zeros((width_small, height_small))
    # weights[350, 250] = 100

    weights_nonzero = np.nonzero(weights)
    values_accumulated = np.zeros((width_small, height_small))
    for idx0, idx1 in tqdm(zip(weights_nonzero[0], weights_nonzero[1]), total=len(weights_nonzero[0])):
        mu = np.array([idx0, idx1])  # maybe order is wrong here, not sure
        gaussian = mvn(mean=mu, cov=(sigma / (down_scale_factor ** 2)))
        values_current = gaussian.pdf(grid)
        values_accumulated += values_current * weights[idx0, idx1]

    values_accumulated = cv2.resize(values_accumulated.copy(), (height, width), interpolation=cv2.INTER_CUBIC)
    values_accumulated /= total
    mean_mask = values_accumulated.transpose((1, 0))

    if mean_mask.max() > 0.0:
        mean_mask /= mean_mask.max()
    mean_mask = (mean_mask * 255).astype("uint8")
    mean_mask = np.repeat(mean_mask[:, :, np.newaxis], 3, axis=2)

    return mean_mask


def get_gaze_distribution(data, config):
    width = 800
    height = 600

    # create the array to accumulate values in
    weights = np.zeros((width, height), dtype=np.float64)
    total = 0

    # loop through the frames
    df_dict = {}
    counter = 0
    for _, row in tqdm(data.iterrows(), total=len(data.index)):
        # load the dataframes if they haven't been loaded yet
        rel_run_path = run_info_to_path(row["subject"], row["run"], row["track_name"])
        if rel_run_path not in df_dict:
            df_screen = pd.read_csv(os.path.join(config["data_root"], rel_run_path, "screen_timestamps.csv"))
            df_gaze = pd.read_csv(os.path.join(config["data_root"], rel_run_path, "gaze_on_surface.csv"))

            # filter out the frames that cannot be found in the index
            indexed_frames = filter_by_property(data, [], {"subject": row["subject"],
                                                           "run": row["run"],
                                                           "track_name": row["track_name"]})["frame"]
            df_screen = df_screen[df_screen["frame"].isin(indexed_frames)]

            # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
            df_gaze = df_gaze[["ts", "frame", "norm_x_su", "norm_y_su"]]
            df_gaze.columns = ["ts", "frame", "x", "y"]
            df_gaze["x"] = np.round(df_gaze["x"] * width).astype(int)
            df_gaze["y"] = np.round((1.0 - df_gaze["y"]) * height).astype(int)

            # filter by screen timestamps
            df_screen, df_gaze = filter_by_screen_ts(df_screen, df_gaze)

            df_dict[rel_run_path] = {"screen": df_screen, "gaze": df_gaze}

        df_gaze = df_dict[rel_run_path]["gaze"]
        current_gaze = df_gaze[df_gaze["frame"] == row["frame"]]
        for _, gaze_row in current_gaze.iterrows():
            x = int(gaze_row["x"])
            y = int(gaze_row["y"])
            if 0 <= x < weights.shape[0] and 0 <= y < weights.shape[1]:
                weights[x, y] += 1
                total += 1
        counter += 1

    # normalise the probability distribution
    if np.sum(weights) > 0.0:
        weights /= np.sum(weights)

    return weights


def generate_from_index(config):
    # load the global index and the split index and combine them (to filter on the global index based on split)
    df_frame_index = pd.read_csv(os.path.join(config["data_root"], "index", "frame_index.csv"))
    df_splits = pd.read_csv(config["split_config"])
    df_frame_index["split"] = df_splits["split"]

    # filter out any frames with no gaze measurements available, not on a valid lap and not on an expected trajectory
    # TODO: maybe simplify this (see compute_statistics.py)
    properties = {
        "gaze_measurement_available": 1,
        "valid_lap": 1,
        "expected_trajectory": 1,
        "split": "train",
    }
    properties.update(config["filter"])
    df_frame_index = filter_by_property(df_frame_index, [], properties)

    # compute the actual mean mask and save the image
    mean_mask = get_mean_mask(df_frame_index, config)
    cv2.imwrite(os.path.join(config["data_root"], "preprocessing_info", "{}.png".format(config["name"])), mean_mask)


def generate_distribution(config):
    # load the global index, but keep everything, since we want the distribution over the entire dataset
    # TODO: might want to change this, but only for one split? e.g. split 11
    df_frame_index = pd.read_csv(os.path.join(config["data_root"], "index", "frame_index.csv"))

    # filter only on gaze_measurement_available for this distribution thing for now
    properties = {
        "gaze_measurement_available": 1,
    }
    properties.update(config["filter"])
    df_frame_index = filter_by_property(df_frame_index, [], properties)

    # do something similar to get_mean_mask to get the distribution and save it as a numpy array
    distribution = get_gaze_distribution(df_frame_index, config)
    np.save(os.path.join(config["data_root"], "preprocessing_info", "{}.npy".format(config["name"])), distribution)


def parse_config(args):
    config = vars(args)
    config["filter"] = {n: v for n, v in config["filter"]}
    try:
        split_index = int(config["split_config"])
        config["split_config"] = os.path.join(config["data_root"], "splits", "split{:03d}.csv".format(split_index))
    except ValueError:
        if config["split_config"].endswith(".json"):
            config["split_config"] = os.path.abspath(config["split_config"])[:-5] + ".csv"
        elif config["split_config"].endswith(".csv"):
            config["split_config"] = os.path.abspath(config["split_config"])
    return config


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument("-md", "--mode", type=str, default="mask", choices=["mask", "distribution"],
                        help="Whether to generate a mask (an 8-bit image) or a distribution (a numpy array that is "
                             "saved with full precision and normalised to sum to 1).")
    PARSER.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    # PARSER.add_argument("-if", "--index_file", type=str, default=None,
    #                     help="CSV file that indexes the frames from which to take the gaze measurements.")
    PARSER.add_argument("-sc", "--split_config", default=0,
                        help="The split of the data to compute statistics for (on the training set). "
                             "Can either be the path to a file or an index.")
    PARSER.add_argument("-f", "--filter", type=pair, nargs="*", default=[],
                        help="Properties and their values to filter by in the format property_name:value.")
    PARSER.add_argument("-n", "--name", type=str, default=None, required=True,
                        help="File name to save the mask under. If left unspecified it will be generated based on "
                             "the filter argument. NOTE: currently needs to be specified to run the script until "
                             "I figure out a good way of specifying the options in the file name or somewhere else.")
    PARSER.add_argument("-dsf", "--down_scale_factor", type=int, default=2,
                        help="Downscale factor to use for generating the mean map.")
    # TODO: should this also be saved with the index files? would be pretty annoying/messy I think
    # I think it makes more sense to compute these (and the resulting videos) once the final splits to use are decided

    # parse the arguments
    ARGS = PARSER.parse_args()
    CONFIG = parse_config(ARGS)

    # main function call
    # generate_mean_mask(ARGS)
    # if CONFIG["split_config"] is not None:
    if CONFIG["mode"] == "mask":
        generate_from_index(CONFIG)
    elif CONFIG["mode"] == "distribution":
        generate_distribution(CONFIG)

    # TODO: if no index file is specified, should in principle split the data ourselves
    # however, this would mean that all options that can be specified for generate_splits should also appear here
    # not sure if this is a very nice way of doing it; an alternative might be to load a config file (e.g. in JSON
    # format) as a dictionary and use the values from that => this could then also be used to dynamically create
    # the same splits (since the seed is specified) when the datasets are created for training
