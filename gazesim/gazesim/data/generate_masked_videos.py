import os
import numpy as np
import cv2
import re

from tqdm import tqdm
from gazesim.data.utils import iterate_directories

PRINTED_FPS_INFO = False


def handle_single_video(config, run_dir, gt_video_path):
    global PRINTED_FPS_INFO

    suffix = "" if config["frames_per_second"] == 60 else "_{}".format(config["frames_per_second"])

    rgb_video_path = os.path.join(run_dir, "{}.mp4".format(config["video_name"]))
    hard_video_path = os.path.join(run_dir, "{}_hard_mask_{}{}.mp4".format(
        config["video_name"], config["ground_truth_name"], suffix))
    soft_video_path = os.path.join(run_dir, "{}_soft_mask_{}{}.mp4".format(
        config["video_name"], config["ground_truth_name"], suffix))
    mean_video_path = os.path.join(run_dir, "{}_{}{}.mp4".format(
        config["video_name"], config["mean_mask_name"], suffix))

    current_output_mode = [om for om in config["output_mode"]]

    if "hard_mask" in current_output_mode and config["skip_existing"] and os.path.isfile(hard_video_path):
        current_output_mode = [om for om in current_output_mode if om != "hard_mask"]
    if "soft_mask" in current_output_mode and config["skip_existing"] and os.path.isfile(soft_video_path):
        current_output_mode = [om for om in current_output_mode if om != "soft_mask"]
    if "mean_mask" in current_output_mode and config["skip_existing"] and os.path.isfile(mean_video_path):
        current_output_mode = [om for om in current_output_mode if om != "mean_mask"]

    if len(current_output_mode) == 0:
        print("Skipping '{}' since there is nothing to write.".format(run_dir))
        return

    # load the mean mask
    mean_mask = None
    if config["mean_mask_path"] is not None:
        mean_mask = cv2.imread(config["mean_mask_path"]).astype("float64")[:, :, 0]
        mean_mask /= 255.0
        if mean_mask.max() > 0.0:
            mean_mask /= mean_mask.max()

    # initialise the capture and writer objects
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    gt_cap = cv2.VideoCapture(gt_video_path)

    # determine whether frames have to be skipped based on the frame rate of the input video
    w, h, fps, fourcc, num_frames = (rgb_cap.get(i) for i in range(3, 8))
    fps = int(fps)
    assert 60 % fps == 0, "FPS needs to be a divisor of 60."
    gt_step = int(60 / fps)

    hard_video_writer, soft_video_writer, mean_video_writer = None, None, None
    if "hard_mask" in current_output_mode:
        hard_video_writer = cv2.VideoWriter(hard_video_path, int(fourcc), fps, (int(w), int(h)), True)
    if "soft_mask" in current_output_mode:
        soft_video_writer = cv2.VideoWriter(soft_video_path, int(fourcc), fps, (int(w), int(h)), True)
    if "mean_mask" in current_output_mode:
        mean_video_writer = cv2.VideoWriter(mean_video_path, int(fourcc), fps, (int(w), int(h)), True)

    if not PRINTED_FPS_INFO and fps != 60:
        print("Input video frame rate determined to be {} FPS (instead of 60 FPS); "
              "using this value for video generation.".format(fps))
        PRINTED_FPS_INFO = True

    """
    # TODO: different check here?
    if num_frames != gt_cap.get(7):
        print("RGB and GT video do not have the same number of frames for {}.".format(run_dir))
        return
    """

    # loop through all frames
    sml = config["soft_masking_lambda"]
    for frame_idx in tqdm(range(int(num_frames))):
        rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        gt_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * gt_step)

        rgb_frame = rgb_cap.read()[1].astype("float64")
        gt_frame = gt_cap.read()[1].astype("float64")[:, :, 0]
        gt_frame /= 255.0
        if gt_frame.max() > 0.0:
            gt_frame /= gt_frame.max()

        if "hard_mask" in current_output_mode:
            hard_masked = rgb_frame * gt_frame[:, :, np.newaxis]
            hard_masked = np.round(hard_masked).astype("uint8")
            hard_video_writer.write(hard_masked)

        if "soft_mask" in current_output_mode:
            soft_masked = sml * rgb_frame + (1 - sml) * rgb_frame * gt_frame[:, :, np.newaxis]
            soft_masked = np.round(soft_masked).astype("uint8")
            soft_video_writer.write(soft_masked)

        if "mean_mask" in current_output_mode:
            mean_masked = rgb_frame * mean_mask[:, :, np.newaxis]
            mean_masked = np.round(mean_masked).astype("uint8")
            mean_video_writer.write(mean_masked)

    if "hard_mask" in current_output_mode:
        hard_video_writer.release()
    if "soft_mask" in current_output_mode:
        soft_video_writer.release()
    if "mean_mask" in current_output_mode:
        mean_video_writer.release()

    print("Finished writing {} for '{}' to '{}'.".format(
        " and ".join(current_output_mode), config["ground_truth_name"], run_dir))


def main(config):
    for run_dir in iterate_directories(config["data_root"], config["track_name"]):
        # need ground-truth to be there
        gt_video_path = os.path.join(run_dir, "{}.mp4".format(config["ground_truth_name"]))

        # check if required file exists
        if os.path.exists(gt_video_path):
            handle_single_video(config, run_dir, gt_video_path)
        else:
            print("Skipping '{}' because no video for '{}' exists.".format(run_dir, config["ground_truth_name"]))


def parse_config(args):
    config = vars(args)
    config["data_root"] = os.path.abspath(config["data_root"])
    config["mean_mask_path"] = None if args.mean_mask_path is None else os.path.abspath(config["mean_mask_path"])
    config["mean_mask_name"] = ""
    if config["mean_mask_path"] is not None:
        config["mean_mask_name"] = os.path.splitext(os.path.basename(config["mean_mask_path"]))[0]
    if "all" in config["output_mode"]:
        config["output_mode"] = ["soft_mask", "hard_mask", "mean_mask"]
    result = re.search(r"flightmare_\d+", config["video_name"])
    if result is not None:
        fps = int(result[0][11:])
        if fps != config["frames_per_second"]:
            print("Input video name indicates that videos have a frame rate of {} FPS, but {} FPS is the "
                  "specified frame rate; using the former.".format(fps, config["frames_per_second"]))
            config["frames_per_second"] = fps
        # TODO: also needs to "accept" "flightmare_mpc_FPS" or something like that at some point
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-vn", "--video_name", type=str, default="screen",
                        help="The name of the input videos.")
    parser.add_argument("-tn", "--track_name", type=str, nargs="+", default=["flat", "wave"],
                        help="The name of the track.")
    parser.add_argument("-gtn", "--ground_truth_name", type=str, default="moving_window_frame_mean_gt",
                        help="The name of the ground-truth video.")
    parser.add_argument("-om", "--output_mode", type=str, nargs="+", default=["soft_mask", "hard_mask"],
                        choices=["all", "soft_mask", "hard_mask", "mean_mask"],
                        help="Which mask type to generate.")
    parser.add_argument("-fps", "--frames_per_second", type=int, default=60,
                        help="The frame rate of the input videos.")
    parser.add_argument("-l", "--soft_masking_lambda", type=float, default=0.2,
                        help="Lambda for soft masking.")
    parser.add_argument("-mmp", "--mean_mask_path", type=str, default=None,
                        help="File path to the mean mask to use for masking.")
    parser.add_argument("-se", "--skip_existing", action="store_true")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    main(parse_config(arguments))

