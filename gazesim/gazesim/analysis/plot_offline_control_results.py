import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style

from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb

style.use("ggplot")

df_offline = pd.read_csv("/home/simon/Desktop/weekly_meeting/meeting23-24/presentation_figures_and_videos/"
                         "control_offline/control_combined.csv")

separate_by_experiment = False
separate_models = True

if separate_models:
    errors = ["throttle", "roll", "pitch", "yaw"]
    error_labels = [e.capitalize() for e in errors]

    models = df_offline["model"].unique()
    model_labels = ["Unmasked", "Mean mask", "Hard mask", "Soft mask", "Dual branch\n(unmasked + hard)"]

    width = 0.6
    colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    colors = colors[:5] + ["#2da690"]

    # first plot Makrigiorgos results
    data = []
    for m in models:
        data.append(df_offline.loc[(df_offline["experiment"] == "replication") & (df_offline["model"] == m),
                                   ["throttle_l1", "roll_l1", "pitch_l1", "yaw_l1"]].values[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=100)
    x_per_err = np.arange(len(errors))
    x_individual = [x_per_err + i * width / len(models) for i in [-2, -1, 0, 1, 2]]
    for model_idx, model_data in enumerate(data):
        ax.bar(
            x_individual[model_idx], model_data, width / 5,
            label=model_labels[model_idx],
            color=colors[model_idx],
        )

    ax.set_xticks(x_per_err)
    ax.set_xticklabels(error_labels)
    ax.set_ylabel("Mean absolute error")
    ax.legend()
    fig.tight_layout()

    # then plot ResNet results
    data = []
    for m in models:
        data.append(df_offline.loc[(df_offline["experiment"] == "resnet") & (df_offline["model"] == m),
                                   ["throttle_l1", "roll_l1", "pitch_l1", "yaw_l1"]].values[0])
    data.append(df_offline.loc[(df_offline["experiment"] == "replication") &
                               (df_offline["model"] == "mean_mask"),
                               ["throttle_l1", "roll_l1", "pitch_l1", "yaw_l1"]].values[0])
    model_labels.append("Best original model")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=100)
    x_per_err = np.arange(len(errors))
    x_individual = [x_per_err + i * width / len(data) for i in [-3, -2, -1, 0, 1, 3]]
    for model_idx, model_data in enumerate(data):
        ax.bar(
            x_individual[model_idx], model_data, width / 6,
            label=model_labels[model_idx],
            color=colors[model_idx],
        )

    ax.set_xticks(x_per_err)
    ax.set_xticklabels(error_labels)
    ax.set_ylabel("Mean absolute error")
    ax.legend()
    fig.tight_layout()

    plt.show()

else:
    if separate_by_experiment:
        test = []
        errors = ["throttle", "roll", "pitch", "yaw"]
        models = df_offline["model"].unique()
        for e in errors:
            test.append(df_offline[f"{e}_l1"].values.reshape(2, -1)[::-1])

        print(test)

        labels = ["Unmasked", "Mean mask", "Hard mask", "Soft mask", "Dual branch\n(unmasked + hard)"]

        original_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = original_colors[:len(errors)]
        colors_hsv = [rgb_to_hsv(to_rgb(c)) for c in colors]
        colors_lighter = [hsv_to_rgb(c * np.array([1.0, 0.6, 1.0])) for c in colors_hsv]

        all_colors = [colors, colors_lighter]

        experiments = df_offline["experiment"].unique()[::-1]

        width_big = 0.4
        width_small = 0.3

        x_err = np.arange(len(models))

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

        x_exp = [x_err + i * width_big / len(experiments) for i in [-1, 1]]
        for exp_idx, exp in enumerate(experiments):
            x_error = [x_exp[exp_idx] + i * width_small / len(errors) for i in [-1.5, -0.5, 0.5, 1.5]]
            for error_idx, error_data in enumerate(test):
                ax.bar(
                    x_error[error_idx], error_data[exp_idx], width_small / len(errors),
                    label=errors[error_idx].capitalize() if exp_idx == 0 else None,
                    color=all_colors[exp_idx][error_idx],
                )

        x_ticks = []
        x_labels = []
        for idx in range(len(labels)):
            x_ticks.append(x_exp[0][idx])
            x_ticks.append(x_exp[1][idx])
            x_labels.append(experiments[0].capitalize())
            x_labels.append(experiments[1].capitalize())

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Mean Absolute Error")
        ax.legend()
        # ax.text(0.02, 0.5, "test1", fontsize=14, transform=plt.gcf().transFigure)
        for l_idx, l in enumerate(labels):
            plt.text(l_idx, -0.03, l.capitalize(), fontsize=11, transform=ax.transData, ha="center", va="center", color="#555555")

        fig.tight_layout()
        plt.show()
    else:
        test = []
        models = []
        for model in df_offline["model"].unique():
            models.append(model)
            test.append(df_offline.loc[df_offline["model"] == model, ["throttle_l1", "roll_l1", "pitch_l1", "yaw_l1"]].values[::-1])

        labels = ["Unmasked", "Mean mask", "Hard mask", "Soft mask", "Dual branch\n(unmasked + hard)"]

        original_colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        original_colors.insert(4, "#2da690")
        colors = original_colors[:len(models)]
        colors_hsv = [rgb_to_hsv(to_rgb(c)) for c in colors]
        colors_lighter = [hsv_to_rgb(c * np.array([1.0, 0.6, 1.0])) for c in colors_hsv]

        all_colors = [colors, colors_lighter]

        errors = ["throttle", "roll", "pitch", "yaw"]
        experiments = df_offline["experiment"].unique()[::-1]

        width_big = 0.4
        width_small = 0.3

        x_err = np.arange(len(errors))

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

        x_exp = [x_err + i * width_big / len(experiments) for i in [-1, 1]]
        for exp_idx, exp in enumerate(experiments):
            x_model = [x_exp[exp_idx] + i * width_small / len(models) for i in [-2, -1, 0, 1, 2]]
            for model_idx, model_data in enumerate(test):
                ax.bar(
                    x_model[model_idx], model_data[exp_idx], width_small / 5,
                    label=labels[model_idx] if exp_idx == 0 else None,
                    color=all_colors[exp_idx][model_idx],
                )

        x_ticks = []
        x_labels = []
        for idx in range(len(errors)):
            x_ticks.append(x_exp[0][idx])
            x_ticks.append(x_exp[1][idx])
            x_labels.append(experiments[0].capitalize())
            x_labels.append(experiments[1].capitalize())

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Mean Absolute Error")
        ax.legend()
        # ax.text(0.02, 0.5, "test1", fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0, -0.03, "Throttle", fontsize=11, transform=ax.transData, ha="center", color="#555555")
        plt.text(1, -0.03, "Roll", fontsize=11, transform=ax.transData, ha="center", color="#555555")
        plt.text(2, -0.03, "Pitch", fontsize=11, transform=ax.transData, ha="center", color="#555555")
        plt.text(3, -0.03, "Yaw", fontsize=11, transform=ax.transData, ha="center", color="#555555")

        fig.tight_layout()
        plt.show()

