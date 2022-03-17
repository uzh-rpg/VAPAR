import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
# import seaborn as sns

from pprint import pprint
from tqdm import tqdm
from gazesim.analysis.utils import extract_tensorboard_data

style.use("ggplot")

EXPERIMENT_NAMES = ["resnet-larger-screen-da-nca", "resnet-larger-screen-mean-mask-da-nca",
                    "resnet-larger-screen-hard-mask-da-nca", "resnet-larger-screen-soft-mask-da-nca",
                    "resnet-larger-dual-branch-screen-da-nca"]
EXPERIMENT_LABELS = ["Raw image", "Mean mask", "Hard mask", "Soft mask", "Dual-branch\n(raw + hard mask)"]

SAVE_NAMES = ["raw", "mean_mask", "hard_mask", "soft_mask", "dual_branch"]

INDIVIDUAL_LOSSES = ["throttle", "roll", "pitch", "yaw"]


def main(config):
    # find matching log folders
    experiment_log_dirs = {en: [] for en in EXPERIMENT_NAMES}
    for log_dir in sorted(os.listdir(config["log_root"])):
        for en in EXPERIMENT_NAMES:
            # result = re.search(en + r"_\d".format(en), log_dir)
            result = re.search(en, log_dir)
            if result is not None:
                pos = result.end() + 1
                run_number = int(log_dir[pos:(pos + 1)])
                experiment_log_dirs[en].append((run_number, os.path.join(config["log_root"], log_dir, "tensorboard")))

    final_epoch = 4

    # "gather" the runs for the different experiments
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=100, sharey=True)
    fig_absolute, ax_absolute = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=100)
    fig_individual, ax_individual = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), dpi=100, sharex=True)

    results = {
        "model": [],
        "total_mse": [],
        "total_l1": [],
        "throttle_l1": [],
        "roll_l1": [],
        "pitch_l1": [],
        "yaw_l1": [],
    }

    last_val_global_step = None
    for e_idx, en in tqdm(enumerate(EXPERIMENT_NAMES), total=len(EXPERIMENT_NAMES), disable=True):
        combined_df = None
        for run_index, run_dir in tqdm(experiment_log_dirs[en], disable=True):
            df = extract_tensorboard_data(run_dir)
            if "loss/val/total/mse" not in df.columns:
                rn = {f"loss/train/output_control/{c}/mse": f"loss/train/{c}/mse" for c in INDIVIDUAL_LOSSES}
                rn.update({f"loss/val/output_control/{c}/mse": f"loss/val/{c}/mse" for c in INDIVIDUAL_LOSSES})
                rn.update({f"loss/val/output_control/{c}/l1": f"loss/val/{c}/l1" for c in INDIVIDUAL_LOSSES})
                rn.update({"loss/val/total": "loss/val/total/mse", "loss/train/total": "loss/train/total/mse"})
                df = df.rename(rn, axis=1)

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

        max_step = np.flatnonzero(combined_df.index == 508670 * final_epoch)[0]
        combined_df = combined_df.iloc[:max_step+1]

        combined_df = combined_df.groupby(combined_df.index).mean()
        combined_df["loss/train/total/mse"] = combined_df["loss/train/total/mse"].rolling(window=200).mean().values
        mask = np.isfinite(combined_df["loss/val/total/mse"])

        test = [np.flatnonzero(combined_df.index == 508670 * i)[0] for i in range(1, 1 + final_epoch)]
        ax[0].plot(combined_df.index, combined_df["loss/train/total/mse"], label=EXPERIMENT_LABELS[e_idx],
                   marker="o", markevery=test, mfc="white", ms=5)
        ax[1].plot(combined_df.index[mask], combined_df.loc[mask, "loss/val/total/mse"], label=EXPERIMENT_LABELS[e_idx],
                   marker="o", markevery=list(range(3 - 1, 3 * final_epoch, 3)), mfc="white", ms=5)

        results["model"].append(SAVE_NAMES[e_idx])
        results["total_mse"].append(combined_df.loc[mask, "loss/val/total/mse"].values[0])

        total_l1_loss = 0
        for l_idx, loss in enumerate(INDIVIDUAL_LOSSES):
            index_0 = l_idx // 2
            index_1 = l_idx % 2
            ax_individual[index_0][index_1].plot(
                combined_df.index[mask],
                combined_df.loc[mask, f"loss/val/{loss}/l1"],
                label=EXPERIMENT_LABELS[e_idx],
                marker="o",
                markevery=list(range(3 - 1, 3 * final_epoch, 3)),
                mfc="white",
                ms=5
            )
            total_l1_loss += combined_df.loc[mask, f"loss/val/{loss}/l1"].values[0]
            results[f"{loss}_l1"].append(combined_df.loc[mask, f"loss/val/{loss}/l1"].values[0])

        results["total_l1"].append(total_l1_loss / 4)

    data = pd.DataFrame(results)
    data.to_csv("/home/simon/Desktop/weekly_meeting/meeting23-24/presentation_figures_and_videos/resnet.csv",
                index=False)

    ax[0].set_xlim(left=0)
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("(Smoothed) mean training loss (MSE)")
    ax[0].set_xticks([508670 * i for i in range(1, 1 + final_epoch)])
    ax[0].set_xticklabels([i for i in range(1, 1 + final_epoch)])
    # ax[0].legend()
    ax[1].set_xlim(left=0)
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Mean test loss (MSE)")
    ax[1].set_xticks([508670 * i for i in range(1, 1 + final_epoch)])
    ax[1].set_xticklabels([i for i in range(1, 1 + final_epoch)])
    ax[1].legend(fancybox=True)
    fig.tight_layout()

    ax_absolute.set_xlim(left=0)
    ax_absolute.set_xlabel("Training steps")
    ax_absolute.set_ylabel("Average total test error (L1) over 3 runs")
    ax_absolute.legend()
    fig_absolute.tight_layout()

    for l_idx, loss in enumerate(INDIVIDUAL_LOSSES):
        index_0 = l_idx // 2
        index_1 = l_idx % 2
        ax_individual[index_0][index_1].set_xlim(left=0)
        ax_individual[index_0][index_1].set_xlabel("Epochs")
        ax_individual[index_0][index_1].set_ylabel(f"Mean {loss} test error (L1)")
        ax_individual[index_0][index_1].set_xticks([508670 * i for i in range(1, 1 + final_epoch)])
        ax_individual[index_0][index_1].set_xticklabels([i for i in range(1, 1 + final_epoch)])
        if l_idx == 0:
            ax_individual[index_0][index_1].legend()

    fig_individual.tight_layout()
    plt.show()


if __name__ == "__main__":
    main({"log_root": os.getenv("GAZESIM_LOG")})


