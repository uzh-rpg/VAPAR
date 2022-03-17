import os
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
from pims import PyAVReaderIndexed
from scipy.stats import multivariate_normal as mvn

########################################
# FUNCTIONS RELATED TO THE FILE SYSTEM #
########################################


def iterate_directories(data_root, track_names=None):
    if track_names is None:
        track_names = ["flat"]

    directories = []
    if re.search(r"/\d\d_", data_root):
        directories.append(data_root)
    elif re.search(r"/s0\d\d", data_root):
        for run in sorted(os.listdir(data_root)):
            run_dir = os.path.join(data_root, run)
            if os.path.isdir(run_dir) and run[3:] in track_names:
                directories.append(run_dir)
    else:
        for subject in sorted(os.listdir(data_root)):
            subject_dir = os.path.join(data_root, subject)
            if os.path.isdir(subject_dir) and subject.startswith("s"):
                for run in sorted(os.listdir(subject_dir)):
                    run_dir = os.path.join(subject_dir, run)
                    if os.path.isdir(run_dir) and run[3:] in track_names:
                        directories.append(run_dir)

    return directories


def parse_run_info(run_dir):
    if run_dir[-1] == "/":
        run_dir = run_dir[:-1]

    # return the subject number, run number and track name
    result = re.search(r"/s0\d\d", run_dir)
    subject_number = None if result is None else int(result[0][2:])

    result = re.search(r"/s0\d\d/\d\d_", run_dir)
    run_number = None if result is None else int(result[0][6:8])

    result = re.search(r"/s0\d\d/\d\d_.+", run_dir)
    track_name = None if result is None else result[0][9:]

    info = {
        "subject": subject_number,
        "run": run_number,
        "track_name": track_name
    }

    return info


def run_info_to_path(subject, run, track_name):
    return os.path.join("s{:03d}".format(subject), "{:02d}_{}".format(run, track_name))


def pair(arg):
    # custom argparse type
    if ":" in arg:
        arg_split = arg.split(":")
        property_name = arg_split[0]
        property_value = arg_split[1]

        try:
            property_value = int(arg_split[1])
        except ValueError:
            try:
                property_value = float(arg_split[1])
            except ValueError:
                pass
    else:
        property_name = arg
        property_value = 1

    return property_name, property_value


def get_indexed_reader(video_path):
    # timed_reader = PyAVReaderTimed(video_path, cache_size=1)
    # num_frames = len(timed_reader)
    cv2_reader = cv2.VideoCapture(video_path)
    num_frames = int(cv2_reader.get(7))

    toc_lengths = [1] * num_frames
    toc_ts = [i for i in range(0, num_frames * 256, 256)]
    toc = {"lengths": toc_lengths, "ts": toc_ts}

    return PyAVReaderIndexed(video_path, toc=toc)


def resolve_split_index_path(split, data_root=None, return_base_name=True):
    try:
        split_index = int(split)
        split = os.path.join("splits", "split{:03d}".format(split_index))
        if data_root is not None:
            split = os.path.join(data_root, split)
    except ValueError:
        if split.endswith(".json"):
            split = os.path.abspath(split)[:-5]
        elif split.endswith(".csv"):
            split = os.path.abspath(split)[:-4]
    if not return_base_name:
        split += ".csv"

    return split


#################################
# FUNCTIONS RELATED TO PLOTTING #
#################################


def generate_gaussian_heatmap(width=800,
                              height=600,
                              mu=np.array([400, 300]),
                              sigma=np.array([[200, 0], [0, 200]]),
                              down_scale_factor=5,
                              combine="max",
                              plot=False,
                              show_progress=False):
    mu_list = []
    if isinstance(mu, np.ndarray):
        if len(mu.shape) == 1:
            mu_list.append(mu)
        elif len(mu.shape) == 2:
            for m in mu:
                mu_list.append(m)
    elif isinstance(mu, list):
        mu_list = mu

    sigma_list = []
    if isinstance(sigma, np.ndarray):
        if len(sigma.shape) == 2:
            sigma_list.append(sigma)
        elif len(sigma.shape) == 3:
            for s in sigma:
                sigma_list.append(s)
    elif isinstance(sigma, list):
        sigma_list = sigma

    if len(mu_list) > 1 and len(sigma_list) == 1:
        sigma_list = [sigma_list[0] for _ in range(len(mu_list))]
    if len(mu_list) == 1 and len(sigma_list) > 1:
        mu_list = [mu_list[0] for _ in range(len(sigma_list))]

    width_small = int(np.round(width / down_scale_factor))
    height_small = int(np.round(height / down_scale_factor))

    grid = np.mgrid[0:width_small, 0:height_small]
    grid = grid.transpose((1, 2, 0))

    values_accumulated = np.zeros((width_small, height_small))
    for m, s in tqdm(zip(mu_list, sigma_list), disable=(not show_progress)):
        gaussian = mvn(mean=(m / down_scale_factor), cov=(s / (down_scale_factor ** 2)))
        values_current = gaussian.pdf(grid)
        values_accumulated = np.maximum(values_accumulated, values_current)

    values_accumulated = cv2.resize(values_accumulated, (height, width), interpolation=cv2.INTER_CUBIC)
    if not values_accumulated.sum() == 0.0:
        values_accumulated /= values_accumulated.sum()
    values_accumulated = values_accumulated.transpose((1, 0))

    if plot:
        plt.imshow(values_accumulated, cmap="jet", interpolation="nearest")
        plt.show()
        """
        values_accumulated = cv2.applyColorMap((values_accumulated / values_accumulated.max() * 255).astype("uint8"),
                                               cv2.COLORMAP_JET)
        cv2.imshow("Heatmap", values_accumulated)
        cv2.waitKey(0)
        """

    return values_accumulated


##########################################################
# FUNCTIONS RELATED TO STANDARD OPERATIONS ON DATAFRAMES #
##########################################################


def filter_by_screen_ts(df_screen, df_other, buffer=(1 / 120)):
    # use only those timestamps that can be matched to the "screen" video
    first_screen_ts = df_screen["ts"].iloc[0] - buffer
    last_screen_ts = df_screen["ts"].iloc[-1] + buffer
    df_other = df_other[(first_screen_ts <= df_other["ts"]) & (df_other["ts"] <= last_screen_ts)].copy()

    # compute timestamp windows around each frame to "sort" the gaze measurements into
    frame_ts_prev = df_screen["ts"].values[:-1]
    frame_ts_next = df_screen["ts"].values[1:]
    frame_ts_midpoint = ((frame_ts_prev + frame_ts_next) / 2).tolist()
    frame_ts_midpoint.insert(0, first_screen_ts - (1 / 120))
    frame_ts_midpoint.append(last_screen_ts + (1 / 120))

    # update the gaze dataframe with the "screen" frames
    for frame_idx, ts_prev, ts_next in zip(df_screen["frame"], frame_ts_midpoint[:-1], frame_ts_midpoint[1:]):
        df_other.loc[(ts_prev <= df_other["ts"]) & (df_other["ts"] < ts_next), "frame"] = frame_idx

    return df_screen, df_other


# TODO: maybe move this to the top of the file (especially if there will be "constants")
PROPERTY_KEEP_DICT = {
    "expected_trajectory": 1,
    "valid_lap": 1,
    "left_turn": 1,
    "right_turn": 1,
    "left_half": 1,
    "right_half": 1
}


def filter_by_property(data, properties, property_keep_dict=None, add_to_properties=False):
    # assume that data is a dataframe in the format of the main index file
    # TODO: maybe just use dict for properties? with the respective value(s) for a key being specified in there
    if property_keep_dict is None:
        property_keep_dict = PROPERTY_KEEP_DICT
    else:
        for k in property_keep_dict:
            if k not in properties:
                properties.append(k)
        if add_to_properties:
            property_keep_dict.update(PROPERTY_KEEP_DICT)

    assert all([p in property_keep_dict.keys() for p in properties]), \
        "All properties to filter on must have a defined condition/value to keep them."
    assert all([p in data.columns for p in properties]), "All properties must be column names of the input dataframes."

    for p in properties:
        # TODO: maybe add different checks for different properties if required (e.g. isin())
        data = data[data[p] == property_keep_dict[p]]

    return data


def filter_by_property_improved(data, properties_and, properties_or=None):
    # assume that data is a dataframe in the format of the main index file
    # properties_and is a dictionary
    # properties_or is a list of dictionaries
    assert all([p in data.columns for p in properties_and]), \
        "All properties (AND) must be column names of the input dataframes."
    assert all([p in data.columns for p in [o for s in properties_or for o in s.keys()]]), \
        "All properties (OR) must be column names of the input dataframes."

    for p in properties_and:
        # TODO: maybe add different checks for different properties if required (e.g. isin())
        data = data[data[p] == properties_and[p]]

    if properties_or is not None:
        for property_set in properties_or:
            match = False
            for prop, value in property_set.items():
                match = match | (data[prop] == value)
            data = data[match]

    return data


"""
import pandas as pd
d = pd.DataFrame({"one": [0, 0, 1, 1, 0, 0, 1, 1], "two": [1, 0, 0, 0, 0, 0, 0, 1], "three": [1, 1, 1, 1, 0, 0, 0, 0]})
p_and = {"three": 1}
p_or = [{"one": 1, "two": 1}]
d = filter_by_property_improved(d, p_and, p_or)
print(d)
"""


def find_contiguous_sequences(data, new_index=False):
    # find contiguous sequences of frames
    sequences = []
    frames = data["frame"]
    index = np.arange(len(frames.index)) if new_index else frames.index
    jumps = (frames - frames.shift()) != 1
    frames = list(index[jumps]) + [index[-1] + 1]
    for i, start_index in enumerate(frames[:-1]):
        # note that the range is exclusive: [start_frame, end_frame)
        sequences.append((start_index, frames[i + 1]))
    return sequences


def fps_reduction_index(data, fps=60, groupby_columns=None, return_sub_index_by_group=False):
    assert 60 % fps == 0, "FPS needs to be a divisor of 60."
    step = int(60 / fps)
    idx_separate = data.groupby(groupby_columns).apply(lambda x: x[::step].index.values).values
    idx_combined = np.concatenate(idx_separate)
    idx_combined_bool = data.index.isin(idx_combined)
    if return_sub_index_by_group:
        sub_idx = []
        for idx in idx_separate:
            sub_idx.extend(list(range(len(idx))))
        return idx_combined_bool, idx_combined, sub_idx
    return idx_combined_bool, idx_combined
