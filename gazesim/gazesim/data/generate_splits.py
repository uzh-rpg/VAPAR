# is there really much of a difference between splitting for stacked frames/non-stacked frames?
# => I think the most important distinction would be that there need to be "enough" stacked-frame samples
#    for individual clips... not sure what I'm even writing but you get the gist

# TODO: a problem that needs to be handled in some way is that there could be overlap between train/val/test
#  "frame stacks" if the splitting is done naively => the easiest way to solve this is to have the smallest
#  "unit of division" for this type of data be e.g. one lap (i.e. a lap is only in one of the sets),
#  otherwise one would have to create somewhat larger clips that don't overlap from which frame stacks
#  are taken (seems pretty complicated and possibly unnecessary)

# maybe try just printing some statistics first
import os
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit as MSSS
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as MSKF
from gazesim.data.utils import filter_by_property_improved, pair


def find_next_split_name(split_dir):
    max_index = -1
    for file in os.listdir(split_dir):
        if os.path.isfile(os.path.join(split_dir, file)) and file.endswith(".csv"):
            file_index = int(file[6:8])
            if file_index > max_index:
                max_index = file_index
    return "split{:03d}".format(max_index + 1)


def find_best_split(d, i, ts, s):
    # TODO: I guess we could still use this but with the multi-class splitting? for now unused
    closest = 10.0
    closest_splits = None
    for rep in range(100):
        tri, tei = train_test_split(i, train_size=ts, random_state=rep, stratify=s)
        test_dist = np.abs((d.loc[d.index[tei]].sum() / d.sum()) - ts)
        if closest_splits is None or test_dist < closest:
            closest = test_dist
            closest_splits = (tri, tei)
    return closest_splits


def ungroup_indices(data, grouped_data, group_by_columns, *indices):
    total_index = np.arange(len(data.index))

    ungrouped_indices = []
    for idx in indices:
        if len(idx) == 0:
            ungrouped_idx = idx
        elif isinstance(idx[0], np.ndarray):
            ungrouped_idx = []
            for sub_idx in idx:
                sub_idx = list(zip(*grouped_data.index[sub_idx].tolist()))
                match = False
                for combination in zip(*sub_idx):
                    current_match = True
                    for col, value in zip(group_by_columns, combination):
                        current_match = current_match & (data[col] == value)
                    match = match | current_match
                sub_idx = total_index[match]
                ungrouped_idx.append(sub_idx)
        else:
            ungrouped_idx = list(zip(*grouped_data.index[idx].tolist()))
            match = False
            for combination in zip(*ungrouped_idx):
                current_match = True
                for col, value in zip(group_by_columns, combination):
                    current_match = current_match & (data[col] == value)
                match = match | current_match
            ungrouped_idx = total_index[match]

        ungrouped_indices.append(ungrouped_idx)

    return tuple(ungrouped_indices)


def generate_splits(data, config, return_data_index=False):
    # assume that the dataframe has the format of the main index file
    # config should contain:
    # split size for train, val, test; if they don't add up to 1, take train as basis and make the others proportional?
    # "stratification level" => will always be track_name > subject > run > lap_index, so maybe just use those numbers?
    #   => then -1 could be no splitting by this stuff
    # actually, this has different meanings depending on what we group by right? (e.g. grouping only by subject...)
    #   => is there actually any use to grouping by "more" than the stratification level and then using that
    #      level? I don't think so, right? actually, this just changes the results depending e.g. on how many "runs"
    #      there are below the "subject" level, basically re-weighting everything based on the number of runs, since
    #      one subject appears essentially #runs times in the index...
    #   => not sure if this might be desirable in some scenarios, but I strongly suspect that it isn't
    stratification_level = config["stratification_level"]
    group_by_level = config["group_by_level"]
    assert stratification_level in [-1, 2, 3], "Invalid stratification level {}.".format(stratification_level)
    assert stratification_level == -1 or stratification_level <= group_by_level, \
        "'Group-by' level ({}) needs to be <= stratification level ({}).".format(group_by_level, stratification_level)
    assert all(data["valid_lap"] == 1), "Only data from valid laps can be split, otherwise laps might " \
                                        "contain too few frames to be split in a stratified manner."

    # copy the dataframe and add the numeric equivalent for "track_name"
    data = data.copy()
    data["track_index"] = pd.factorize(data["track_name"])[0]

    # determine the split sizes and relative split size of validation and test split
    split_size_sum = config["train_size"] + config["val_size"] + config["test_size"]
    if split_size_sum != 1.0:
        config["train_size"] /= split_size_sum
        config["val_size"] /= split_size_sum
        config["test_size"] /= split_size_sum
    rel_val_size = config["val_size"]
    if (config["val_size"] + config["test_size"]) != 0.0:
        rel_val_size = config["val_size"] / (config["val_size"] + config["test_size"])

    if config["train_size"] == 1.0:
        train_index = np.arange(len(data.index))
        val_index = np.array([], dtype=int)
        test_index = np.array([], dtype=int)
        if return_data_index:
            return train_index, val_index, test_index, data.index.values
        return train_index, val_index, test_index

    group_by_columns = ["track_index", "subject", "run", "lap_index"]
    if stratification_level == -1:
        # basically use all levels for stratification, but without any grouping
        index = np.arange(len(data.index))
        strata = ""
        for col_idx, col in enumerate(group_by_columns):
            strata += data[col].astype(str)
            if col_idx < len(group_by_columns) - 1:
                strata += "_"

        # create multiple indices for cross validation if cv_folds > 1
        if config["cv_splits"] == 1:
            # first split into train and rest, then the rest into validation and test set
            train_index, val_test_index = train_test_split(index, train_size=config["train_size"], stratify=strata,
                                                           random_state=config["random_seed"])
            if rel_val_size != 1.0:
                val_index, test_index = train_test_split(val_test_index, train_size=rel_val_size,
                                                         stratify=strata.iloc[val_test_index],
                                                         random_state=config["random_seed"])
            else:
                val_index = val_test_index
                test_index = np.array([], dtype=int)
        else:
            cv_generator = StratifiedKFold(n_splits=config["cv_splits"], random_state=config["random_seed"])
            train_index = []
            val_index = None
            test_index = []
            for tr_idx, te_idx in cv_generator.split(index, strata):
                tr_idx = index[tr_idx]
                te_idx = index[te_idx]

                train_index.append(tr_idx)
                test_index.append(te_idx)
    else:
        # group by the specified columns and create index and strata for the resulting dataframe
        # group_by_columns = group_by_columns
        grouped_data = data.groupby(group_by_columns[:group_by_level]).count()["frame"]

        index = np.arange(len(grouped_data.index))
        strata = np.array([idx[:stratification_level] for idx in grouped_data.index.values])

        if config["cv_splits"] == 1:
            train_index, val_test_index = list(MSSS(n_splits=1, test_size=None, train_size=config["train_size"],
                                                    random_state=config["random_seed"]).split(index, strata))[0]
            if rel_val_size != 1.0:
                val_index, test_index = list(MSSS(n_splits=1, test_size=None, train_size=rel_val_size,
                                                  random_state=config["random_seed"])
                                             .split(val_test_index, strata[val_test_index]))[0]
                val_index = index[val_test_index[val_index]]
                test_index = index[val_test_index[test_index]]
            else:
                val_index = index[val_test_index]
                test_index = np.array([], dtype=int)
            train_index = index[train_index]

            train_index, val_index, test_index = ungroup_indices(data, grouped_data, group_by_columns,
                                                                 train_index, val_index, test_index)
        else:
            cv_generator = MSKF(n_splits=config["cv_splits"], random_state=config["random_seed"])
            train_index = []
            val_index = None
            test_index = []
            for tr_idx, te_idx in cv_generator.split(index, strata):
                tr_idx = index[tr_idx]
                te_idx = index[te_idx]

                train_index.append(tr_idx)
                test_index.append(te_idx)

            train_index, test_index = ungroup_indices(data, grouped_data, group_by_columns, train_index, test_index)

        # convert from the "grouped dataframe" index to the overall data index
        """
        total_index = np.arange(len(data.index))

        train_index = list(zip(*grouped_data.index[train_index].tolist()))
        match = False
        for combination in zip(*train_index):
            current_match = True
            for col, value in zip(group_by_columns, combination):
                current_match = current_match & (data[col] == value)
            match = match | current_match
        train_index = total_index[match]

        if val_index.size != 0:
            val_index = list(zip(*grouped_data.index[val_index].tolist()))
            match = False
            for combination in zip(*val_index):
                current_match = True
                for col, value in zip(group_by_columns, combination):
                    current_match = current_match & (data[col] == value)
                match = match | current_match
            val_index = total_index[match]

        if test_index.size != 0:
            test_index = list(zip(*grouped_data.index[test_index].tolist()))
            match = False
            for combination in zip(*test_index):
                current_match = True
                for col, value in zip(group_by_columns, combination):
                    current_match = current_match & (data[col] == value)
                match = match | current_match
            test_index = total_index[match]
        """

    if return_data_index:
        return train_index, val_index, test_index, data.index.values
    return train_index, val_index, test_index


def create_split_index(config):
    # TODO: to be called when this script is executed stand-alone; should save the splits in the indexing directory then
    #  by specifying which split each frame belongs to (including none of them/invalid, i.e. save for EVERY frame to
    #  match the global index)
    #  => save with the split size and the stratification level/type in the name of the file
    #  => should there be some separate mode for stacked frames? I think it's just a matter of stratification level...

    # 1. parse the arguments and construct the config
    # 2. load the global index
    # 3. remove the invalid laps
    # 4. return the indices (including the dataframe index of the filtered data)
    # 5. for the new dataframe, set everything to "none"/"invalid" and then index the
    #    dataframe index with the split indices to set them to the correct value
    # 6. save to CSV file

    # TODO: check that the exact same parameters have not been generated already
    # e.g. some function check_redundant()

    # load the global index dataframes
    frame_index_path = os.path.join(config["data_root"], "index", "frame_index.csv")
    gaze_gt_path = os.path.join(config["data_root"], "index", "gaze_gt.csv")
    control_gt_path = os.path.join(config["data_root"], "index", "control_gt.csv")
    df_frame_index = pd.read_csv(frame_index_path)
    df_gaze_gt = pd.read_csv(gaze_gt_path).drop(["frame", "subject", "run"], axis=1)
    df_control_gt = pd.read_csv(control_gt_path).drop(["frame", "subject", "run"], axis=1)
    df_frame_index = pd.concat([df_frame_index, df_gaze_gt, df_control_gt], axis=1)
    # TODO: different states as well? so far handled by drone_measurement_available, which might be different for MPC?

    # create the dataframe to save the split information into and populate it with "none" values
    df_split = df_frame_index[["frame"]].copy()
    df_split.columns = ["split"]
    df_split["split"] = "none"

    # filter out the invalid laps, otherwise problems can arise when trying to split in a stratified manner
    # df_frame_index = df_frame_index[df_frame_index["valid_lap"] == 1]
    properties = {
        "rgb_available": 1,
        "valid_lap": 1,
        "expected_trajectory": 1,
    }
    properties.update(config["filter"])
    df_frame_index = filter_by_property_improved(df_frame_index, properties, config["filter_or"])

    # create the splits
    train_index, val_index, test_index, data_index = generate_splits(df_frame_index, config, return_data_index=True)

    if config["cv_splits"] == 1:
        # use the data index to determine the actual positions of the split indices
        train_index = data_index[train_index]
        val_index = data_index[val_index]
        test_index = data_index[test_index]

        # fill the dataframe with values indicating the splits
        df_split.iloc[train_index, 0] = "train"
        df_split.iloc[val_index, 0] = "val"
        df_split.iloc[test_index, 0] = "test"
    else:
        # repeat the same as above for each split and store them in different columns
        for s_idx, (tr_idx, te_idx) in enumerate(zip(train_index, test_index)):
            df_split[f"split_{s_idx}"] = "none"

            tr_idx = data_index[tr_idx]
            te_idx = data_index[te_idx]

            df_split.iloc[tr_idx, s_idx + 1] = "train"
            df_split.iloc[te_idx, s_idx + 1] = "test"

        df_split = df_split.drop("split", axis=1)

    # determine the name of the file to save the splits into and save the dataframe to CSV
    split_info = {k: config[k] for k in config.keys()
                  & {"train_size", "val_size", "test_size", "cv_splits", "stratification_level", "group_by_level",
                     "random_seed", "filter", "filter_or"}}

    split_dir = os.path.join(config["data_root"], "splits")
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    split_name = find_next_split_name(split_dir)
    split_path = os.path.join(config["data_root"], "splits", split_name + ".csv")
    split_info_path = os.path.join(config["data_root"], "splits", split_name + "_info.json")

    df_split.to_csv(split_path, index=False)
    with open(split_info_path, "w") as f:
        json.dump(split_info, f)


def parse_config(args):
    # this is here for now in case anything changes
    # => maybe the conversion to sum to 1.0 should be done here?
    #    but then using the function stand-alone wouldn't check that anymore
    config = vars(args)
    config["filter"] = {n: v for n, v in config["filter"]}
    config["filter_or"] = [{n: v for n, v in property_set} for property_set in config["filter_or"]]
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # command line arguments
    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-trs", "--train_size", type=float, default=0.7,
                        help="The relative size of the training split.")
    parser.add_argument("-vs", "--val_size", type=float, default=0.15,
                        help="The relative size of the validation split.")
    parser.add_argument("-tes", "--test_size", type=float, default=0.15,
                        help="The relative size of the test split.")
    parser.add_argument("-cv", "--cv_splits", type=int, default=1,
                        help="Number of cross validation folds (if cv_folds=1, only create a single "
                             "train/test/val split). Since the data should only be split into two folds for "
                             "each individual split, the parameters for train/test/val set size are ignored.")
    parser.add_argument("-sl", "--stratification_level", type=int, default=-1, choices=[-1, 2, 3, 4],
                        help="Decides how the splits are generated in a stratified manner by trying to divide data "
                             "across the different splits as evenly as possible based on the frame properties "
                             "'track_name', 'subject', 'run', and 'lap_index' (in that order). The level refers to "
                             "the number of these that are used to define a 'class'; if -1 is chosen, frames from "
                             "the same lap can be put into different splits, otherwise the splits are 'cleanly "
                             "separated' by subject/run.")
    parser.add_argument("-gbl", "--group_by_level", type=int, default=1, choices=[1, 2, 3, 4],
                        help="TODO")
    parser.add_argument("-rs", "--random_seed", type=int, default=112,
                        help="The random seed to use for generating the splits.")
    parser.add_argument("-f", "--filter", type=pair, nargs="+", default=[],
                        help="Properties and their values to filter by in the format property_name:value.")
    parser.add_argument("-fo", "--filter_or", type=pair, nargs="+", default=[], action="append",
                        help="Properties and their values to filter by in the format property_name:value. "
                             "Uses OR instead of AND (-f/--filter) as condition. Multiple OR conditions "
                             "can be specified by repeating the flag.")

    """
    parser.add_argument("-cm", "--compose_manually", action="store_true",
                        help="If not specified, default index files are loaded and data is filtered "
                             "on some default columns. When specified --index_files, --out_name, and "
                             "--rename_splits should/can be specified.")
    parser.add_argument("-cm", "--compose_manually", action="store_true",
                        help="If not specified, default index files are loaded and data is filtered "
                             "on some default columns. When specified --index_files, --out_name, and "
                             "--rename_splits should/can be specified.")
    # TODO: should maybe just have functionality to select some data w/o any splitting and just have a 0-1 index
    # TODO: maybe option to only do rgb_available "split"?
    """

    # parse the arguments
    arguments = parser.parse_args()

    # call the function thingy
    create_split_index(parse_config(arguments))

