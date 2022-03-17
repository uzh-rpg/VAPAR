import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("ggplot")

df_offline = pd.read_csv("/home/simon/Desktop/weekly_meeting/meeting23-24/presentation_figures_and_videos/"
                         "results_attention_offline/attention_combined.csv")

df_att = df_offline.loc[df_offline["model"].str.contains("_att")]
df_gaze = df_offline.loc[df_offline["model"].str.contains("_gaze")]

plot_att = True

models = ["resnet", "deep_sup", "high_res", "mean", "random"]
model_labels = ["ResNet", "Deep supervision", "High-res", "Mean", "Random"]

if plot_att:
    errors = ["kl", "cc", "time_per_sample"]
    error_labels = ["KL-divergence\n(lower is better)", "Pearson Correlation Coefficient\n(higher is better)",
                    "Inference time per sample\n(lower is better)"]

    fig, ax = plt.subplots(ncols=3, figsize=(12, 5), dpi=100)
    for e_idx, e in enumerate(errors):
        for m_idx, m in enumerate(models):
            data = df_att.loc[(df_att["model"] == f"{m}_att") & (df_att["error"] == e), "value"].values[0]
            data = data if data != np.inf else 0
            ax[e_idx].bar(m_idx, data, label=model_labels[m_idx])
        ax[e_idx].set_ylabel(error_labels[e_idx])
        ax[e_idx].get_xaxis().set_ticks([])
    ax[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    ax[0].text(0.9, 0.03, "Inf.", fontsize=11, transform=ax[0].transAxes, ha="center", va="center", color="#555555")
    ax[2].text(0.7, 0.03, "0", fontsize=11, transform=ax[2].transAxes, ha="center", va="center", color="#555555")
    ax[2].text(0.9, 0.03, "0", fontsize=11, transform=ax[2].transAxes, ha="center", va="center", color="#555555")

    fig.tight_layout()
    plt.show()
else:
    errors = ["total_l1", "partial_l1_x", "partial_l1_y"]
    error_labels = ["Mean Absolute Error (total)", "Mean Absolute Error (x)", "Mean Absolute Error (y)"]

    fig, ax = plt.subplots(ncols=3, figsize=(12, 5), dpi=100)
    for e_idx, e in enumerate(errors):
        for m_idx, m in enumerate(models):
            data = df_gaze.loc[(df_gaze["model"] == f"{m}_gaze") & (df_gaze["error"] == e), "value"].values[0]
            data = data if data != np.inf else 0
            ax[e_idx].bar(m_idx, data, label=model_labels[m_idx])
        ax[e_idx].set_ylabel(error_labels[e_idx])
        ax[e_idx].get_xaxis().set_ticks([])
    ax[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    plt.show()

