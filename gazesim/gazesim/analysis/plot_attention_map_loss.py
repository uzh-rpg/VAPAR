import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from gazesim.models.utils import image_log_softmax, image_softmax
from gazesim.models.resnet import ResNet18BaseModelSimple
from gazesim.data.old_datasets import get_dataset

# TODO: plot the loss for each frame (where GT is available) on a plot with the x-axis being frames
# mark whether parts are left/right turn, whether it is a valid lap, whether it follows the expected trajectory

# 1. need to loop through all videos with screen_frame_info.csv (otherwise no input)
# 2. for that video, need to loop through all frames and for those with GT available, compute the loss
# 3. then need to plot the loss...


def extract_ones_sequences(sequence):
    # identify locations where shifting by 1 gives result != 0
    shifted = np.roll(sequence, 1)
    shifted[0] = sequence[0]
    diff = sequence - shifted

    # get the indices and add to them if necessary
    indices = list(np.where(diff != 0)[0])
    if 0 not in indices and sequence[0] == 1:
        indices = [0] + indices
    if len(indices) == 0:
        return []
    if sequence[indices[-1]] == 1:
        indices = indices + [len(sequence)]

    # make list of the start and end indices of the sequences of 1s
    start_end_indices = []
    for i in range(0, len(indices), 2):
        start_end_indices.append((indices[i], indices[i + 1]))

    return start_end_indices


def handle_single_video(args, run_dir, frame_info_path, gt_video_path):
    # model info/parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_info = torch.load(args.model_path, map_location=device)

    # creating model
    # TODO: should work with other model classes
    model = ResNet18BaseModelSimple(transfer_weights=False)
    model.load_state_dict(model_info["model_state_dict"])
    model = model.to(device)
    model.eval()

    # defining the loss function(s)
    loss_function = torch.nn.KLDivLoss()

    # load the dataframe with the info
    df_frame_info = pd.read_csv(frame_info_path)

    # create the dataset
    video_dataset = get_dataset(run_dir, data_type="single_video", resize_height=300, use_pims=args.use_pims)

    # loop through all frames in frame_info
    losses = []
    gt_available = []
    valid_lap = []
    expected_trajectory = []
    turn_left = []
    turn_right = []
    for _, row in tqdm(df_frame_info.iterrows(), total=len(df_frame_info.index)):
        # if ground-truth is available, compute the loss, otherwise insert nan
        if row["gt_available"] == 1:
            frame, label = video_dataset[row["frame"]]
            frame, label = frame.unsqueeze(0).to(device), label.unsqueeze(0).to(device)
            predicted_label = model(frame)
            loss = loss_function(image_log_softmax(predicted_label), label).item()

            # put loss in the list of losses
            losses.append(loss)
        else:
            losses.append(np.nan)

        gt_available.append(row["gt_available"])
        valid_lap.append(row["valid_lap"])
        expected_trajectory.append(row["expected_trajectory"])
        turn_left.append(row["turn_left"])
        turn_right.append(row["turn_right"])

        if row["frame"] == 2000:
            break

    frames = df_frame_info["frame"].values[:2001]
    losses = np.array(losses)
    losses = np.convolve(losses, np.ones((args.window_size,)) / args.window_size, mode="same")
    gt_available = np.array(gt_available)
    valid_lap = np.array(valid_lap)
    expected_trajectory = np.array(expected_trajectory)
    turn_left = np.array(turn_left)
    turn_right = np.array(turn_right)
    mask = np.isfinite(losses)

    """
    fig, host = plt.subplots()
    temp = host.twinx()

    host.plot(frames[mask], losses[mask])
    temp.plot(frames[other_mask], valid_lap[other_mask])

    temp.set_ylim(-1, 2)
    """
    gt_available_indices = extract_ones_sequences(gt_available)
    valid_lap_indices = extract_ones_sequences(valid_lap)
    expected_trajectory_indices = extract_ones_sequences(expected_trajectory)
    turn_left_indices = extract_ones_sequences(turn_left)
    turn_right_indices = extract_ones_sequences(turn_right)

    fig, ax = plt.subplots(figsize=(15, 5))
    handles = []
    labels = []
    h, = ax.plot(frames[mask], losses[mask], label="KL div. loss")
    handles.append(h)
    labels.append("KL div. loss")
    for i, (start, end) in enumerate(turn_left_indices):
        h, = ax.plot((start, end), (0, 0), color="c", linewidth=2, label="Left turn")
        if i == 0:
            handles.append(h)
            labels.append("Left turn")
    for i, (start, end) in enumerate(turn_right_indices):
        h, = ax.plot((start, end), (-0.1E-5, -0.1E-5), color="m", linewidth=2, label="Right turn")
        if i == 0:
            handles.append(h)
            labels.append("Right turn")
    for i, (start, end) in enumerate(expected_trajectory_indices):
        h, = ax.plot((start, end), (-0.2E-5, -0.2E-5), color="g", linewidth=2, label="Exp. traj.")
        if i == 0:
            handles.append(h)
            labels.append("Exp. traj.")
    for i, (start, end) in enumerate(valid_lap_indices):
        h, = ax.plot((start, end), (-0.3E-5, -0.3E-5), color="r", linewidth=2, label="Valid lap")
        if i == 0:
            handles.append(h)
            labels.append("Valid lap")
    for i, (start, end) in enumerate(gt_available_indices):
        h, = ax.plot((start, end), (-0.4E-5, -0.4E-5), color="k", linewidth=2, label="GT available")
        if i == 0:
            handles.append(h)
            labels.append("GT available")
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # TODO: add second y-axis to plot these on


def handle_single_run(args, run_dir):
    # need screen_frame_info.csv for information about valid lap etc.
    # need ground-truth to be there as well
    gt_video_path = os.path.join(run_dir, f"{args.ground_truth_name}.mp4")
    df_frame_info_path = os.path.join(run_dir, "screen_frame_info.csv")

    # check if required files exist
    if os.path.exists(gt_video_path) and os.path.exists(df_frame_info_path):
        handle_single_video(args, run_dir, df_frame_info_path, gt_video_path)


def main(args):
    args.data_root = os.path.abspath(args.data_root)
    # loop through directory structure and create plots for every run/video that has the necessary information
    # check if data_root is already a subject or run directory
    if re.search(r"/s0\d\d", args.data_root):
        if re.search(r"/\d\d_", args.data_root):
            handle_single_run(args, args.data_root)
        else:
            for run in sorted(os.listdir(args.data_root)):
                run_dir = os.path.join(args.data_root, run)
                if os.path.isdir(run_dir) and args.track_name in run_dir:
                    handle_single_run(args, run_dir)
    else:
        for subject in sorted(os.listdir(args.data_root)):
            subject_dir = os.path.join(args.data_root, subject)
            if os.path.isdir(subject_dir):
                for run in sorted(os.listdir(subject_dir)):
                    run_dir = os.path.join(subject_dir, run)
                    if os.path.isdir(run_dir) and args.track_name in run_dir:
                        handle_single_run(args, run_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-tn", "--track_name", type=str, default="flat",
                        help="The name of the track.")
    parser.add_argument("-gtn", "--ground_truth_name", type=str, default="moving_window_gt",
                        help="The name of the ground-truth video.")
    parser.add_argument("-ws", "--window_size", type=int, default=30,
                        help="Size of the window over which a moving average is compute to smooth the plot.")
    parser.add_argument("--use_pims", action="store_true",
                        help="Whether to use PIMS (PyAV) instead of OpenCV for reading frames.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    main(arguments)
