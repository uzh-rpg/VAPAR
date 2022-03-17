# TODO: go through all laps (similar to generate_expected_trajectory_entries.py), check whether it is valid and
#  an expected trajectory and the print the time it took together with its "identifiers" (run_dir, lap_index)

# TODO: given some run_dir and lap_index, simply plot the trajectory from above for visual inspection


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pprint import pprint
from gazesim.data.utils import iterate_directories


def load_data(directory):
    inpath_drone = os.path.join(directory, "drone.csv")
    inpath_laps = os.path.join(directory, "laptimes.csv")
    inpath_exp_traj = os.path.join(directory, "expected_trajectory.csv")

    if not os.path.isfile(inpath_exp_traj):
        return None

    df_drone = pd.read_csv(inpath_drone)
    df_laps = pd.read_csv(inpath_laps)
    df_exp_traj = pd.read_csv(inpath_exp_traj)

    trajectories = {}
    for index, row in df_laps.iterrows():
        if row["is_valid"] == 1 and row["lap"] in df_exp_traj["lap"] \
                and df_exp_traj.loc[df_exp_traj["lap"] == row["lap"], "expected_trajectory"].values[0] == 1:
            temp = df_drone[df_drone["ts"].between(row["ts_start"], row["ts_end"])]

            if len(temp.index) > 0:
                trajectories[row["lap"]] = {
                    "x": temp["PositionX"].values,
                    "y": temp["PositionY"].values,
                    # "z": temp["PositionZ"].values,
                    "t": row["lap_time"],
                }

            # print(row)
            # print(df_exp_traj)
            # print(df_exp_traj["lap"] == row["lap"])
            # print(df_exp_traj.loc[df_exp_traj["lap"] == row["lap"], "lap"].values[0])
            # exit()

    return trajectories


def evaluate_all_trajectories(args):
    data = []
    for run_dir in tqdm(iterate_directories(args.data_root, track_names=args.track_name), disable=False):
        trajectories = load_data(run_dir)

        if trajectories is not None:
            # print(trajectories)
            for lap_index, lap_data in trajectories.items():
                data.append((lap_data["t"], run_dir, lap_index, lap_data["x"], lap_data["y"]))

    data = sorted(data, key=lambda d: d[0], reverse=True)
    times = np.array([d[0] for d in data])

    print("Laps to consider:", len(data))

    percentiles = [0, 25, 37, 50, 67, 75, 90, 95, 100]
    percentile_values = np.percentile(times, percentiles, interpolation="nearest")
    percentile_indices = [np.where(times == pv)[0] for pv in percentile_values]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    for p_idx, p in enumerate(percentiles):
        d = data[int(percentile_indices[p_idx])]
        print("Percentile {}:".format(p))
        print(d[:3])

        plt.plot(d[3], d[4], alpha=0.8, label=p)  # color="#c2c2c2",

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("tight")
    # ax.axis("off")
    ax.set_xlim(-37.14723358154297, 37.202500915527345)
    ax.set_ylim(-19.70142822265625, 21.74341659545899)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-tn", "--track_name", type=str, default="flat",
                        help="The name of the track.")

    arguments = parser.parse_args()

    evaluate_all_trajectories(arguments)

