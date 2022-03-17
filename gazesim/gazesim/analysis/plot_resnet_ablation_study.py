import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

from pprint import pprint
from tqdm import tqdm
from gazesim.analysis.utils import extract_tensorboard_data

style.use("ggplot")
plt.rcParams.update({"font.size": 11})

EXPERIMENT_NAMES = ["resnet-screen", "resnet-screen-vel-angvel", "resnet-screen-vel", "resnet-image-only", "state-only"]
EXPERIMENT_LABELS = ["Image + full state", "Image + vel. + ang. vel.", "Image + vel.", "Image only", "Full state only"]

INDIVIDUAL_LOSSES = ["throttle", "roll", "pitch", "yaw"]

MARKERS = ["v", "^", "<", ">", "o"]


def main(config):
    # find matching log folders
    experiment_log_dirs = {en: [] for en in EXPERIMENT_NAMES}
    for log_dir in sorted(os.listdir(config["log_root"])):
        for en in EXPERIMENT_NAMES:
            result = re.search(r"{}-\d".format(en), log_dir)
            if result is not None:
                pos = result.end() - 1
                run_number = int(log_dir[pos:(pos + 1)])
                experiment_log_dirs[en].append((run_number, os.path.join(config["log_root"], log_dir, "tensorboard")))

    # "gather" the runs for the different experiments
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=100, sharey=True)
    fig_absolute, ax_absolute = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=100)
    fig_individual, ax_individual = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), dpi=100, sharex=True)

    last_val_global_step = None
    for e_idx, en in tqdm(enumerate(EXPERIMENT_NAMES), total=len(EXPERIMENT_NAMES)):
        combined_df = None
        for run_index, run_dir in tqdm(experiment_log_dirs[en], disable=True):
            df = extract_tensorboard_data(run_dir)

            # only keep datasets up to the last validation run
            if last_val_global_step is None:
                last_val_global_step = df.index[~df["loss/val/total/mse"].isna()][-1]
            df = df.loc[:last_val_global_step]

            # add columns to show which model and which run the data is from
            df["experiment"] = en
            df["run"] = run_index

            # concatenate dataframes
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.concat([combined_df, df])

        combined_df = combined_df.groupby(combined_df.index).mean()
        combined_df["loss/train/total/mse"] = combined_df["loss/train/total/mse"].rolling(window=200).mean().values
        mask = np.isfinite(combined_df["loss/val/total/mse"])

        test = [np.flatnonzero(combined_df.index == 508670 * i)[0] for i in range(1, 11)]
        ax[0].plot(combined_df.index, combined_df["loss/train/total/mse"], label=EXPERIMENT_LABELS[e_idx],
                   marker="o", markevery=test, mfc="white", ms=5)
        ax[1].plot(combined_df.index[mask], combined_df.loc[mask, "loss/val/total/mse"], label=EXPERIMENT_LABELS[e_idx],
                   marker="o", markevery=list(range(0, 10)), mfc="white", ms=5)

        ax_absolute.plot(combined_df.index[mask], combined_df.loc[mask, "loss/val/total/l1"], label=EXPERIMENT_LABELS[e_idx])
        ax_absolute.grid()

        for l_idx, loss in enumerate(INDIVIDUAL_LOSSES):
            index_0 = l_idx // 2
            index_1 = l_idx % 2
            ax_individual[index_0][index_1].plot(
                combined_df.index[mask],
                combined_df.loc[mask, f"loss/val/{loss}/l1"],
                label=EXPERIMENT_LABELS[e_idx],
                marker="o",
                markevery=list(range(0, 10)),
                mfc="white",
                ms=5
            )

    ax[0].set_xlim(left=0)
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("(Smoothed) total training loss (MSE)")
    ax[0].set_xticks([508670 * i for i in range(1, 11)])
    ax[0].set_xticklabels([i for i in range(1, 11)])
    # ax[0].legend()
    ax[1].set_xlim(left=0)
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Total test loss (MSE)")
    ax[1].set_xticks([508670 * i for i in range(1, 11)])
    ax[1].set_xticklabels([i for i in range(1, 11)])
    ax[1].legend(fancybox=True)
    fig.tight_layout()

    ax_absolute.set_xlim(left=0)
    ax_absolute.set_xlabel("Epochs")
    ax_absolute.set_ylabel("Total test error (L1)")
    ax_absolute.set_xticks([508670 * i for i in range(1, 11)])
    ax_absolute.set_xticklabels([i for i in range(1, 11)])
    ax_absolute.legend()
    fig_absolute.tight_layout()

    for l_idx, loss in enumerate(INDIVIDUAL_LOSSES):
        index_0 = l_idx // 2
        index_1 = l_idx % 2
        ax_individual[index_0][index_1].set_xlim(left=0)
        ax_individual[index_0][index_1].set_xlabel("Epochs")
        ax_individual[index_0][index_1].set_ylabel(f"{loss.capitalize()} test error (L1)")
        ax_individual[index_0][index_1].set_xticks([508670 * i for i in range(1, 11)])
        ax_individual[index_0][index_1].set_xticklabels([i for i in range(1, 11)])
        if l_idx == len(INDIVIDUAL_LOSSES) - 1:
            ax_individual[index_0][index_1].legend()

    fig_individual.tight_layout()
    plt.show()


if __name__ == "__main__":
    main({"log_root": os.getenv("GAZESIM_LOG")})


