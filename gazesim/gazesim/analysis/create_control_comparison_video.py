import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import cv2
import torch

from collections import OrderedDict
from pprint import pprint
from torch.utils.data._utils.collate import default_collate as to_batch
from tqdm import tqdm
from gazesim.data.utils import find_contiguous_sequences, resolve_split_index_path, run_info_to_path, fps_reduction_index
from gazesim.training.config import parse_config as parse_train_config
from gazesim.training.helpers import resolve_model_class, resolve_dataset_class, resolve_optimiser_class
from gazesim.training.utils import to_device
from gazesim.training.helpers import resolve_losses, resolve_output_processing_func, resolve_logger_class

# TODO options for which "frames" to include
# - include only those with GT given
# - include only those on valid lap, expected trajectory, left/right turn etc.
# - split non-adjacent sections into separately videos/clips (maybe a bit overkill?)

# TODO: need new way to extract clips from index files, maybe:
# - supply index file and only allow validation and test data (set which to take or both)
# - if specified also filter by subject, run, lap
# - extract clips using the existing function for that
# - how do datasets have to be modified? => subindex?
# - everything should be similarly flexible, based on config? model checkpoint?
style.use("ggplot")
plt.rcParams.update({"font.size": 11})


def create_frame(fig, ax, control_gt, control_prediction, input_images):
    # plot in numpy and convert to opencv-compatible image
    error = control_gt - control_prediction
    x = np.arange(len(control_gt))
    x_labels = ["throttle", "roll", "pitch", "yaw"]
    width = 0.3

    ax.bar(x - width, control_gt, width, label="GT", color="#55a868")
    ax.bar(x, control_prediction, width, label="pred", color="#4c72b0")
    ax.bar(x + width, error, width, label="error", color="#c44e52")
    ax.set_ylim(-1, 1)
    # ax.set_xlim(x[0] - 2 * width, x[-1] + 2 * width)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.axhline(c="k", lw=0.2)
    ax.legend()
    # fig.tight_layout()

    fig.canvas.draw()
    plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # stack the frames/labels for the video
    frame = np.hstack(input_images + [plot_image])
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    plt.cla()

    return frame


def main(config):
    # load frame_index and split_index
    frame_index = pd.read_csv(os.path.join(config["data_root"], "index", "frame_index.csv"))
    split_index = pd.read_csv(config["split_config"] + ".csv")
    # TODO: maybe check that the split index actually contains data that can be used properly

    # use GPU if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(config["gpu"])
                          if use_cuda and config["gpu"] < torch.cuda.device_count() else "cpu")

    # define paths
    log_dir = os.path.abspath(os.path.join(os.path.dirname(config["model_load_path"]), os.pardir))
    save_dir = os.path.join(log_dir, "visualisations")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config_path = os.path.join(log_dir, "config.json")

    # load the config
    with open(config_path, "r") as f:
        train_config = json.load(f)
    train_config["data_root"] = config["data_root"]
    train_config["gpu"] = config["gpu"]
    train_config["model_load_path"] = config["model_load_path"]
    train_config["drone_state_names"] = ["vel", "acc", "omega"]
    train_config = parse_train_config(train_config)
    train_config["split_config"] = config["split_config"]
    train_config["input_video_names"] = [config["video_name"]]
    # train_config["frames_per_second"] = 60

    # from pprint import pprint
    # train_config["model_info"] = None
    # pprint(train_config)
    # exit()

    replace_old_keys = {
        "image_conv_": "image_net_conv.",
        "image_fc_": "image_net_fc.",
        "state_fc_": "state_net.",
        # "control_fc_": "control_fc.",
    }

    # load the model
    # print(train_config["drone_state_names"])
    old_model = False
    model_info = train_config["model_info"]
    if old_model and "codevilla" in train_config["model_name"]:
        new_state_dict = OrderedDict()
        # print(train_config["model_info"]["model_state_dict"].keys())
        for key, value in model_info["model_state_dict"].items():
            did_something = False
            for k, v in replace_old_keys.items():
                if k in key:
                    new_state_dict[key.replace(k, v)] = value
                    did_something = True
            if not did_something and "control_fc" not in key and "control_net" not in key:
                new_state_dict[key] = value

            """
            if "image_conv_" in key:
                new_state_dict[key.replace("image_conv_", "image_net_conv.")] = value
            elif "state_fc_" in key:
                new_state_dict[key.replace("state_fc_", "state_fc.")] = value
            else:
                new_state_dict[key] = value
            """
    else:
        new_state_dict = model_info["model_state_dict"]

    model = resolve_model_class(train_config["model_name"])(train_config)
    # pprint(model.state_dict().keys())
    # model.load_state_dict(model_info["model_state_dict"])
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    # TODO: when loading train_config, should replace data_root

    # TODO: apply filter (can include split, subject, run I guess, not sure that lap would make sense)
    #  => actually, if we wanted to just extract a single video for one lap, it would probably make sense...
    # but if the default is that everything is used... then there is a bit of an issue for labeling complete sequences
    # could either ignore that if it is the case, complain/skip if there is overlap in a single sequence or just
    # use a more "rigorous" structure, where we always filter by split => probably the latter

    # taking care of frame rate subsampling stuff
    subsampling_index, subsampling_index_numeric, new_frame_index = fps_reduction_index(
        frame_index, fps=train_config["frames_per_second"],
        groupby_columns=["track_name", "subject", "run"],
        return_sub_index_by_group=True
    )
    frame_index["frame_original_fps"] = frame_index["frame"].copy()
    frame_index["frame"] = -1
    frame_index.loc[subsampling_index_numeric, "frame"] = new_frame_index

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    throttle_gt, roll_gt, pitch_gt, yaw_gt = [], [], [], []
    throttle_pred, roll_pred, pitch_pred, yaw_pred = [], [], [], []
    for split in config["split"]:
        sub_index = (split_index["split"] == split) & subsampling_index
        current_frame_index = frame_index.loc[sub_index]

        # current_frame_index = frame_index.loc[split_index["split"] == split]
        current_dataset = resolve_dataset_class(train_config["dataset_name"])(train_config, split=split)
        current_dataset.return_original = True
        # TODO: contiguous sequences need frame_skip thingy
        sequences = find_contiguous_sequences(current_frame_index, new_index=True)
        # TODO: actually, should probably let this do its thing and

        video_writer_dict = {}
        # TODO: maybe would be better to additionally sort by subject/run somehow and take care of all of the
        #  sequences for an individual run_dir at once
        run_dirs = [run_info_to_path(current_frame_index["subject"].iloc[si],
                                     current_frame_index["run"].iloc[si],
                                     current_frame_index["track_name"].iloc[si])
                    for si, _ in sequences]
        if len(config["subjects"]) > 0:
            check = [current_frame_index["subject"].iloc[si] in config["subjects"] for si, _ in sequences]
            sequences = [sequences[i] for i in range(len(sequences)) if check[i]]
            run_dirs = [run_dirs[i] for i in range(len(run_dirs)) if check[i]]
        if len(config["runs"]) > 0:
            check = [current_frame_index["run"].iloc[si] in config["runs"] for si, _ in sequences]
            sequences = [sequences[i] for i in range(len(sequences)) if check[i]]
            run_dirs = [run_dirs[i] for i in range(len(run_dirs)) if check[i]]
        if len(config["laps"]) > 0:
            check = [current_frame_index["lap_index"].iloc[si] in config["laps"] for si, _ in sequences]
            sequences = [sequences[i] for i in range(len(sequences)) if check[i]]
            run_dirs = [run_dirs[i] for i in range(len(run_dirs)) if check[i]]

        run_dir = run_dirs[0]
        for (start_index, end_index), current_run_dir in tqdm(zip(sequences, run_dirs), disable=False, total=len(sequences)):
            # TODO: can't use frame_index here (which has the original indexing)
            """
            new_run_dir = run_info_to_path(current_frame_index["subject"].iloc[start_index],
                                           current_frame_index["run"].iloc[start_index],
                                           current_frame_index["track_name"].iloc[start_index])
            """
            if current_run_dir != run_dir:
                video_writer_dict[run_dir].release()
                run_dir = current_run_dir

            if run_dir not in video_writer_dict:
                video_dir = os.path.join(save_dir, run_dir)
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)
                video_capture = cv2.VideoCapture(os.path.join(config["data_root"], run_dir,
                                                              "{}.mp4".format(config["video_name"])))
                fps, fourcc = (video_capture.get(i) for i in range(5, 7))
                video_name = "control_comparison_{}_{}.mp4".format(config["video_name"], split)
                video_writer = cv2.VideoWriter(os.path.join(video_dir, video_name), int(fourcc), fps, (1600, 600), True)
                video_writer_dict[run_dir] = video_writer

            for index in tqdm(range(start_index, end_index), disable=False):
                # read the current data sample
                sample = to_batch([current_dataset[index]])
                sample = to_device(sample, device)

                # compute the loss
                prediction = model(sample)
                prediction["output_control"] = resolve_output_processing_func("output_control")(prediction["output_control"])
                """
                individual_losses = torch.nn.functional.mse_loss(prediction["output_control"],
                                                                 sample["output_control"],
                                                                 reduction="none")
                """
                # TODO: maybe plot the difference in some way as well...

                # get the values as numpy arrays
                control_gt = sample["output_control"].cpu().detach().numpy().reshape(-1)
                control_prediction = prediction["output_control"].cpu().detach().numpy().reshape(-1)
                # control_gt /= np.array([20.0, 6.0, 6.0, 6.0])
                # control_prediction /= np.array([20.0, 6.0, 6.0, 6.0])

                # get the input images (for now there will only be one)
                input_images = []
                for key in sorted(sample["original"]):
                    if key.startswith("input_image"):
                        input_images.append(sample["original"][key].cpu().detach().numpy().squeeze())
                        # print(sample["original"][key].cpu().detach().numpy().shape)

                # create the new frame and write it
                frame = create_frame(fig, ax, control_gt, control_prediction, input_images)
                for _ in range(config["slow_down_factor"]):
                    video_writer_dict[run_dir].write(frame)

                throttle_gt.append(control_gt[0])
                roll_gt.append(control_gt[1])
                pitch_gt.append(control_gt[2])
                yaw_gt.append(control_gt[3])

                throttle_pred.append(control_prediction[0])
                roll_pred.append(control_prediction[1])
                pitch_pred.append(control_prediction[2])
                yaw_pred.append(control_prediction[3])

        for _, vr in video_writer_dict.items():
            vr.release()

        fig_show, ax_show = plt.subplots(nrows=4, ncols=1, figsize=(10, 5), dpi=100, sharex=True)

        line = ax_show[0].plot(np.arange(len(throttle_gt)) / 60.0, throttle_pred, label="Throttle (pred)")
        ax_show[0].plot(np.arange(len(throttle_gt)) / 60.0, throttle_gt, label="Throttle (GT)",
                        color=line[0].get_color(), linestyle="--")
        ax_show[0].set_ylim(bottom=-0.1, top=1.1)
        ax_show[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        # ax_show[0].legend()

        ax_show[1]._get_lines.get_next_color()
        line = ax_show[1].plot(np.arange(len(roll_gt)) / 60.0, roll_pred, label="Roll (pred)")
        ax_show[1].plot(np.arange(len(roll_gt)) / 60.0, roll_gt, label="Roll (GT)",
                        color=line[0].get_color(), linestyle="--")
        ax_show[1].set_ylim(bottom=-1.0, top=1.0)
        ax_show[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        # ax_show[1].legend()

        ax_show[2]._get_lines.get_next_color()
        ax_show[2]._get_lines.get_next_color()
        line = ax_show[2].plot(np.arange(len(pitch_gt)) / 60.0, pitch_pred, label="Pitch (pred)")
        ax_show[2].plot(np.arange(len(pitch_gt)) / 60.0, pitch_gt, label="Pitch (GT)",
                        color=line[0].get_color(), linestyle="--")
        ax_show[2].set_ylim(bottom=-1.0, top=1.0)
        ax_show[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        # ax_show[2].legend()

        ax_show[3]._get_lines.get_next_color()
        ax_show[3]._get_lines.get_next_color()
        ax_show[3]._get_lines.get_next_color()
        line = ax_show[3].plot(np.arange(len(yaw_gt)) / 60.0, yaw_pred, label="Yaw (pred)")
        ax_show[3].plot(np.arange(len(yaw_gt)) / 60.0, yaw_gt, label="Yaw (GT)",
                        color=line[0].get_color(), linestyle="--")
        ax_show[3].set_ylim(bottom=-1.0, top=1.0)
        ax_show[3].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        # ax_show[3].legend()

        ax_show[3].set_xlabel("Time [s]")

        fig_show.tight_layout()
        plt.show()


def parse_config(args):
    config = vars(args)
    config["split_config"] = resolve_split_index_path(config["split_config"], config["data_root"])
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-m", "--model_load_path", type=str, default=None,
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-vn", "--video_name", type=str, default="screen",
                        help="The name of the input video.")
    parser.add_argument("-s", "--split", type=str, nargs="+", default=["val"], choices=["train", "val", "test"],
                        help="Splits for which to create videos.")
    parser.add_argument("-sc", "--split_config", type=str, default=0,
                        help="TODO.")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="The GPU to use.")
    parser.add_argument("-sub", "--subjects", type=int, nargs="*", default=[],
                        help="Subjects to use.")
    parser.add_argument("-run", "--runs", type=int, nargs="*", default=[],
                        help="Runs to use.")
    parser.add_argument("-lap", "--laps", type=int, nargs="*", default=[],
                        help="Laps to use.")
    parser.add_argument("-f", "--filter", type=str, default=None, choices=["turn_left", "turn_right"],
                        help="'Property' by which to filter frames (only left/right turn for now).")
    parser.add_argument("-tn", "--track_name", type=str, default="flat",
                        help="The name of the track.")
    parser.add_argument("-gtn", "--ground_truth_name", type=str, default="moving_window_gt",
                        help="The name of the ground-truth video.")
    parser.add_argument("-om", "--output_mode", type=str, default="overlay_maps",
                        choices=["overlay_maps", "overlay_all", "overlay_none", "overlay_raw"],
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-sf", "--slow_down_factor", type=int, default=1,
                        help="Factor by which the output video is slowed down (frames are simply saved multiple times).")
    parser.add_argument("--use_pims", action="store_true",
                        help="Whether to use PIMS (PyAV) instead of OpenCV for reading frames.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    # main(arguments)
    main(parse_config(arguments))

