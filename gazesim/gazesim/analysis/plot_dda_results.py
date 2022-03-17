import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style

from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb

style.use("ggplot")

df_default = pd.read_csv("/home/simon/Desktop/weekly_meeting/meeting23-24/presentation_figures_and_videos/dda_default/dda_default.csv")
df_ablation = pd.read_csv("/home/simon/Desktop/weekly_meeting/meeting23-24/presentation_figures_and_videos/dda_ablation/dda_ablation.csv")
df_generalisation = pd.read_csv("/home/simon/Desktop/weekly_meeting/meeting23-24/presentation_figures_and_videos/dda_generalisation/dda_generalisation.csv")

switch = "ablation"
# switch = "generalisation"
# switch = "default"

models = ["Baseline", "DDA", "AttIn", "AttBr"]
model_labels = ["Baseline (State-Only)", "DDA (Feature Tracks)", "Attention-Input", "Attention-Branching"]

if switch == "default":
    errors = ["dist", "gates"]
    error_labels = ["Mean Flight Distance [m]", "Percentage of gates passed"]

    ablations = ["NoRef", "ReducedState", "NoState"]
    ablation_labels = ["No reference input", "Reduced state", "No state input"]

    width = 0.6

    fig, ax = plt.subplots(ncols=2, figsize=(8, 5), dpi=100)
    for e_idx, e in enumerate(errors):
        x_pos = np.arange(1)
        x_pos_model = [x_pos + i * width / len(models) for i in [-1.5, -0.5, 0.5, 1.5]]
        print(x_pos_model)
        for m_idx, m in enumerate(models):
            data = df_default.loc[(df_default["model"] == f"{m}"), e].values / (10 * 100 if e == "gates" else 1)
            ax[e_idx].bar(x_pos_model[m_idx], data, width / len(models), label=model_labels[m_idx])
        ax[e_idx].set_ylabel(error_labels[e_idx])
        ax[e_idx].get_xaxis().set_ticks([])
    ax[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    plt.show()
elif switch == "ablation":
    errors = ["dist", "gates"]
    error_labels = ["Mean Flight Distance [m]", "Percentage of gates passed"]

    ablations = ["NoRef", "ReducedState", "NoState"]
    ablation_labels = ["No reference input", "Reduced state", "No state input"]

    width = 0.6

    fig, ax = plt.subplots(ncols=2, figsize=(13, 5), dpi=100)
    for e_idx, e in enumerate(errors):
        x_pos = np.arange(len(ablations))
        x_pos_model = [x_pos + i * width / len(models) for i in [-1.5, -0.5, 0.5, 1.5]]
        for m_idx, m in enumerate(models):
            data = df_ablation.loc[(df_ablation["model"] == f"{m}"), e].values / (10 / 100 if e == "gates" else 1)
            ax[e_idx].bar(x_pos_model[m_idx], data, width / len(models), label=model_labels[m_idx])
        ax[e_idx].set_xlabel("Ablation")
        ax[e_idx].set_ylabel(error_labels[e_idx])

        x_ticks = []
        x_labels = []
        for idx in range(len(ablation_labels)):
            x_ticks.append(x_pos[idx])
            x_labels.append(ablation_labels[idx])

        ax[e_idx].set_xticks(x_ticks)
        ax[e_idx].set_xticklabels(x_labels)

    ax[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    plt.show()
elif switch == "generalisation":
    errors = ["dist", "gates"]
    error_labels = ["Mean Flight Distance [m]", "Percentage of gates passed"]

    generalisations = ["DifferentTrajectories", "DifferentTrack", "MultipleLaps"]
    ablation_labels = ["Different trajectories", "Different track", "Multiple laps\n(70 gates)"]

    width = 0.6

    fig, ax = plt.subplots(ncols=2, figsize=(13, 5), dpi=100)
    for e_idx, e in enumerate(errors):
        x_pos = np.arange(len(generalisations))
        x_pos_model = [x_pos + i * width / len(models) for i in [-1.5, -0.5, 0.5, 1.5]]
        for m_idx, m in enumerate(models):
            data = df_generalisation.loc[(df_generalisation["model"] == f"{m}"), e].values / (np.array([10, 10, 70]) / 100 if e == "gates" else 1)
            ax[e_idx].bar(x_pos_model[m_idx], data, width / len(models), label=model_labels[m_idx])
        ax[e_idx].set_xlabel("Generalization")
        ax[e_idx].set_ylabel(error_labels[e_idx])

        x_ticks = []
        x_labels = []
        for idx in range(len(ablation_labels)):
            x_ticks.append(x_pos[idx])
            x_labels.append(ablation_labels[idx])

        ax[e_idx].set_xticks(x_ticks)
        ax[e_idx].set_xticklabels(x_labels)

    ax[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    plt.show()
