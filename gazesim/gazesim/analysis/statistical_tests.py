import os
import pandas as pd

from scipy.stats import chi2_contingency
from gazesim.data.utils import resolve_split_index_path

# TODO: chi-square test
# load the frame index
# load the ground-truth indices
# combine these dataframes
# create new columns that are somehow relevant:
# - whether a sample/frame/row is "valid" or not
# - what category a sample/frame/row belongs to (for now just track x half)


def chi_square(config):
    # TODO: for now this is just a hard-coded chi-square test between categories of validity and track x half
    #  combinations; would be nice to make this more flexible, e.g. by dynamically combining specified columns
    #  into categories (e.g. those that are binary, take the column name, otherwise take the values)

    # first get additional columns from the other "meta-index" CSV files
    df_idx = pd.read_csv(os.path.join(config["data_root"], "index", "frame_index.csv"))
    df_control = pd.read_csv(os.path.join(config["data_root"], "index", "control_gt.csv"))
    df_gaze = pd.read_csv(os.path.join(config["data_root"], "index", "gaze_gt.csv"))
    for c in df_control.columns:
        if c not in df_idx:
            df_idx[c] = df_control[c]
    for c in df_gaze.columns:
        if c not in df_idx:
            df_idx[c] = df_gaze[c]
    if config["split_config"] is not None:
        df_split = pd.read_csv(os.path.join(config["split_config"]) + ".csv")
        df_idx["split"] = df_split["split"]

    # then create the columns of interest
    if config["split_config"] is None:
        index_column = "valid"
        df_idx["valid"] = ((df_idx["valid_lap"] == 1)
                           & (df_idx["expected_trajectory"] == 1)
                           & (df_idx["rgb_available"] == 1)
                           & (df_idx["drone_control_frame_mean_gt"] == 1))
    else:
        index_column = "split"

    df_idx["half"] = "none"
    df_idx.loc[(df_idx["left_half"] == 1), "half"] = "left"
    df_idx.loc[(df_idx["right_half"] == 1), "half"] = "right"
    df_idx["class"] = df_idx["track_name"] + "_" + df_idx["half"]

    # get the frequency table and the results of the chi-square test
    frequency_table = pd.crosstab(df_idx[index_column], df_idx["class"])
    chi2, p, dof, ex = chi2_contingency(frequency_table.values, correction=False)

    print("Frequency table:")
    print(frequency_table)
    print("\nChi-squared results:")
    print("chi^2 = {}".format(chi2))
    print("p-value = {}".format(p))


def resolve_test_function(test):
    if test == "chi2":
        return chi_square


def parse_config(args):
    config = vars(args)
    if config["split_config"] is not None:
        config["split_config"] = resolve_split_index_path(config["split_config"], config["data_root"])
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-sc", "--split_config", type=str, default=None,
                        help="TODO.")
    parser.add_argument("-t", "--test", type=str, default="chi2", choices=["chi2"],
                        help="Which test(s) to perform.")
    arguments = parser.parse_args()

    resolve_test_function(arguments.test)(parse_config(arguments))
