import os
import json
import numpy as np
import pandas as pd
import matplotlib.style as style
import cv2
import torch

from tqdm import tqdm
from gazesim.data.utils import find_contiguous_sequences, resolve_split_index_path, run_info_to_path
from gazesim.data.datasets import ImageDataset
from gazesim.training.config import parse_config as parse_train_config
from gazesim.training.helpers import resolve_model_class, resolve_dataset_class
from gazesim.training.utils import to_device, to_batch
from gazesim.training.helpers import resolve_output_processing_func
from gazesim.models.utils import image_softmax

style.use("ggplot")


def main(config):
    # "formatting"
    output_size = (800, 600)  # TODO: change the other stuff too
    positions = {}
    if config["output_mode"] == "overlay_maps":
        output_size = (800 * 2, 600 + 100)
        positions["ground_truth"] = (10, 40)
        positions["prediction"] = (10, 75)
        positions["valid_lap"] = (310, 40)
        positions["expected_trajectory"] = (310, 75)
        positions["turn_left"] = (610, 40)
        positions["turn_right"] = (610, 75)
    elif config["output_mode"] == "overlay_all":
        output_size = (800, 600 + 300)
        positions["ground_truth"] = (10, 40)
        positions["prediction"] = (10, 75)
        positions["valid_lap"] = (10, 140)
        positions["expected_trajectory"] = (10, 175)
        positions["turn_left"] = (10, 240)
        positions["turn_right"] = (10, 275)
    elif config["output_mode"] == "overlay_none":
        output_size = (800 * 3, 600 + 100)
        positions["ground_truth"] = (410, 75)
        positions["prediction"] = (810, 75)
        positions["valid_lap"] = (10, 40)
        positions["expected_trajectory"] = (10, 75)
        positions["turn_left"] = (410, 40)
        positions["turn_right"] = (810, 40)
    elif config["output_mode"] == "overlay_simple":
        output_size = (800, 600 + 100)
        positions["ground_truth"] = (10, 40)
        positions["prediction"] = (10, 75)
    elif config["output_mode"] == "prediction_only":
        output_size = (800, 600)

    # load frame_index and split_index
    frame_index = pd.read_csv(os.path.join(config["data_root"], "index", "frame_index.csv"))
    split_index = pd.read_csv(config["split_config"] + ".csv")

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
    train_config = parse_train_config(train_config)
    train_config["split_config"] = config["split_config"]
    train_config["input_video_names"] = [config["video_name"]]
    train_config["return_original"] = True

    # load the model
    model_info = train_config["model_info"]
    model = resolve_model_class(train_config["model_name"])(train_config)
    model.load_state_dict(model_info["model_state_dict"])
    model = model.to(device)
    model.eval()

    gaze_model = "gaze" in train_config["model_name"]
    output_name = "output_attention"
    if gaze_model:
        output_name = "output_gaze"

    gaze_data = None
    if gaze_model and config["save_gaze"]:
        gaze_data = {
            "x_gt": [],
            "y_gt": [],
            "x_pred": [],
            "y_pred": [],
        }

    prediction_only = config["output_mode"] == "prediction_only"
    dataset_class = ImageDataset if prediction_only else resolve_dataset_class(train_config["dataset_name"])

    for split in config["split"]:
        current_frame_index = frame_index.loc[split_index["split"] == split]
        current_dataset = dataset_class(train_config, split=split)
        sequences = find_contiguous_sequences(current_frame_index, new_index=True)

        video_writer_dict = {}
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
            if current_run_dir != run_dir:
                if gaze_model and config["save_gaze"]:
                    if gaze_data is not None:
                        gaze_data = pd.DataFrame(gaze_data)
                        gaze_data.to_csv(os.path.join(config["data_root"], run_dir, "gaze_data.csv"))

                    gaze_data = {
                        "x_gt": [],
                        "y_gt": [],
                        "x_pred": [],
                        "y_pred": [],
                    }

                video_writer_dict[run_dir].release()
                run_dir = current_run_dir

            if run_dir not in video_writer_dict:
                video_dir = os.path.join(save_dir, run_dir)
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)
                video_capture = cv2.VideoCapture(os.path.join(config["data_root"], run_dir, "{}.mp4".format(config["video_name"])))
                fps, fourcc = (video_capture.get(i) for i in range(5, 7))
                video_name = "{}_comparison_{}_{}_{}.mp4".format("gaze_gt" if gaze_model else "default_attention_gt",
                                                                 config["video_name"], config["output_mode"], split)
                video_writer = cv2.VideoWriter(os.path.join(video_dir, video_name), int(fourcc), fps, output_size, True)
                video_writer_dict[run_dir] = video_writer

            for index in tqdm(range(start_index, end_index), disable=True):
                # read the current data sample
                sample = to_batch([current_dataset[index]])
                sample = to_device(sample, device)

                # compute the loss
                prediction = model(sample)
                if gaze_model:
                    prediction[output_name] = resolve_output_processing_func(output_name)(prediction[output_name])
                else:
                    # final_func = torch.sigmoid if train_config["losses"][output_name] == "ice" else image_softmax
                    output_processing_func = {
                        "kl": image_softmax,
                        "ice": torch.sigmoid,
                        "mse": lambda x: x,
                    }[train_config["losses"][output_name]]
                    if isinstance(prediction[output_name], dict):
                        prediction[output_name] = output_processing_func(prediction[output_name]["final"])
                    else:
                        prediction[output_name] = output_processing_func(prediction[output_name])

                # get the values as numpy arrays
                attention_gt = None if prediction_only else sample[output_name].cpu().detach().numpy().squeeze()
                attention_prediction = prediction[output_name].cpu().detach().numpy().squeeze()

                # if specified, save GT predictions to CSV
                if gaze_model and config["save_gaze"]:
                    save_gt = attention_gt.copy()
                    save_pred = attention_prediction.copy()

                    # get normalised image coordinates
                    if hasattr(current_dataset, "output_scaling") and current_dataset.output_scaling:
                        save_gt /= np.array([800.0, 600.0])
                        save_pred /= np.array([800.0, 600.0])

                    # store stuff for saving later
                    gaze_data["x_gt"].append(save_gt[0])
                    gaze_data["y_gt"].append(save_gt[1])
                    gaze_data["x_pred"].append(save_pred[0])
                    gaze_data["y_pred"].append(save_pred[1])

                # get the original frame as a numpy array (also convert color for OpenCV)
                frame = sample["original"]["input_image_0"].cpu().detach().numpy().squeeze()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # stack greyscale labels to become RGB
                if not gaze_model:
                    if not prediction_only:
                        attention_gt = np.repeat(attention_gt[np.newaxis, :, :], 3, axis=0).transpose((1, 2, 0))
                    attention_prediction = np.repeat(attention_prediction[np.newaxis, :, :], 3, axis=0).transpose((1, 2, 0))

                    # normalise using the maximum of the maximacap of either label and convert to [0, 255] scale
                    norm_max = attention_prediction.max() if prediction_only else max([attention_gt.max(), attention_prediction.max()])
                    if norm_max != 0:
                        if not prediction_only:
                            attention_gt /= norm_max
                        attention_prediction /= norm_max
                    if not prediction_only:
                        attention_gt = (attention_gt * 255).astype("uint8")
                    attention_prediction = (attention_prediction * 255).astype("uint8")

                    # set all but one colour channel for GT and predicted labels to 0
                    if not prediction_only:
                        attention_gt[:, :, 0] = 0
                        attention_gt[:, :, -1] = 0
                    attention_prediction[:, :, :-1] = 0

                    # scale the attention maps to the right size
                    if not prediction_only:
                        attention_gt = cv2.resize(attention_gt, (800, 600))
                    attention_prediction = cv2.resize(attention_prediction, (800, 600))

                # stack the frames/labels for the video
                temp = np.zeros(output_size[::-1] + (3,), dtype="uint8")
                if config["output_mode"] == "overlay_maps":
                    combined_labels = cv2.addWeighted(attention_gt, 0.5, attention_prediction, 0.5, 0)
                    new_frame = np.hstack((frame, combined_labels))
                    temp[100:, :, :] = new_frame
                elif config["output_mode"] == "overlay_all":
                    frame = np.repeat(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis], 3, axis=2)
                    combined_labels = cv2.addWeighted(attention_gt, 0.5, attention_prediction, 0.5, 0)
                    new_frame = cv2.addWeighted(frame, 0.4, combined_labels, 0.6, 0)
                    temp[300:, :, :] = new_frame
                elif config["output_mode"] == "overlay_none":
                    new_frame = np.hstack((frame, attention_gt, attention_prediction))
                    temp[100:, :, :] = new_frame
                elif config["output_mode"] == "overlay_simple":
                    # right now only overlay_simple "supported"
                    # frame = np.repeat(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis], 3, axis=2)
                    if gaze_model:
                        if hasattr(current_dataset, "output_scaling") and current_dataset.output_scaling:
                            attention_gt /= 2.0
                            attention_prediction /= 2.0
                            attention_gt += np.array([400.0, 300.0])
                            attention_prediction += np.array([400.0, 300.0])
                            # attention_gt[1] = 600.0 - attention_gt[1]
                            # attention_prediction[1] = 600.0 - attention_prediction[1]
                        else:
                            attention_gt = ((attention_gt + 1.0) / 2.0) * np.array([800.0, 600.0])
                            attention_prediction = ((attention_prediction + 1.0) / 2.0) * np.array([800.0, 600.0])
                            # attention_gt[1] = 600.0 - attention_gt[1]
                            # attention_prediction[1] = 600.0 - attention_prediction[1]

                        attention_gt = tuple(np.round(attention_gt).astype(int))
                        attention_prediction = tuple(np.round(attention_prediction).astype(int))

                        frame = cv2.circle(frame, attention_gt, 10, (255, 0, 0), -1)
                        new_frame = cv2.circle(frame, attention_prediction, 10, (0, 0, 255), -1)
                    else:
                        combined_labels = cv2.addWeighted(attention_gt, 0.5, attention_prediction, 1.0, 0)
                        new_frame = cv2.addWeighted(frame, 0.5, combined_labels, 1.0, 0)

                        """
                        test_gt = scipy.ndimage.center_of_mass(attention_gt[:, :, 0])
                        test_pred = scipy.ndimage.center_of_mass(attention_prediction[:, :, -1])

                        test_gt = (int(test_gt[1]), int(test_gt[0]))
                        test_pred = (int(test_pred[1]), int(test_pred[0]))

                        print(test_gt)
                        print(test_pred)

                        new_frame = cv2.circle(new_frame, test_gt, 5, (255, 0, 0), -1)
                        new_frame = cv2.circle(new_frame, test_pred, 5, (0, 0, 255), -1)

                        cv2.imshow("", new_frame)
                        cv2.waitKey(0)
                        """

                    temp[100:, :, :] = new_frame
                elif config["output_mode"] == "prediction_only":
                    # frame = np.repeat(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis], 3, axis=2)
                    # new_frame = cv2.addWeighted(frame, 0.4, attention_prediction, 0.6, 0)
                    new_frame = cv2.add(frame, attention_prediction)
                    temp = new_frame
                new_frame = temp

                # add the other information we want to display
                if not prediction_only:
                    cv2.putText(new_frame, "Ground-truth", positions["ground_truth"], cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 1)
                    cv2.putText(new_frame, "Prediction", positions["prediction"], cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)

                # write the image (possibly multiple times for slowing the video down)
                for _ in range(config["slow_down_factor"]):
                    video_writer_dict[run_dir].write(new_frame)

        for _, vr in video_writer_dict.items():
            vr.release()

        if gaze_model and config["save_gaze"] and gaze_data is not None:
            gaze_data = pd.DataFrame(gaze_data)
            gaze_data.to_csv(os.path.join(save_dir, run_dir, "gaze_data.csv"))


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
    parser.add_argument("-sc", "--split_config", type=str, default=11,
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
    parser.add_argument("-om", "--output_mode", type=str, default="overlay_simple",
                        choices=["overlay_maps", "overlay_all", "overlay_none", "overlay_simple", "prediction_only"],
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-sf", "--slow_down_factor", type=int, default=1,
                        help="Factor by which the output video is slowed down (frames are simply saved multiple times).")
    parser.add_argument("-sgz", "--save_gaze", action="store_true",
                        help="Whether to save gaze predictions or not.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    main(parse_config(arguments))
