import os
import numpy as np
import torch
import time
import scipy.ndimage

from tqdm import tqdm
from gazesim.data.utils import resolve_split_index_path
from gazesim.models.utils import image_softmax
from gazesim.training.helpers import resolve_dataset_class
from gazesim.training.utils import to_device, to_batch, load_model


def measure_inference_time(config):
    model, model_config = load_model(config["model_load_path"], config["gpu"], return_config=True)

    # use GPU if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(config["gpu"])
                          if use_cuda and config["gpu"] < torch.cuda.device_count() else "cpu")

    # get the model name (there could be duplicates in theory, would have to change if one wanted to test that)
    model_name = model_config["model_name"]

    # move model to correct device
    model.to(device)
    model.eval()

    # modify model config for dataset loading
    model_config["data_root"] = config["data_root"]
    model_config["split_config"] = config["split_config"]

    # also store information about the load path
    model_config["eval_model_load_path"] = config["model_load_path"]

    # need dataset to load original videos (for input) and ground-truth
    dataset = resolve_dataset_class(model_config["dataset_name"])(model_config, split=config["split"], training=False)

    # first loop over dataset to initialise all video readers (?)
    temp, output = None, None
    gaze_pos = None
    """
    for s in tqdm(dataset):
        temp = s
    """
    activation_func = None
    if config["compute_gaze"]:
        activation_func = {
            "kl": image_softmax,
            "ice": torch.sigmoid,
            "mse": lambda x: x,
        }[model_config["losses"]["output_attention"]]

    # run the main loop inside a while loop
    iterations = 0
    times = []
    while iterations < config["iterations"]:
        for sample in tqdm(dataset):
            batch = to_device(to_batch([sample]), device)

            if not config["compute_gaze"]:
                start = time.time()
                other_temp = model(batch)
                end = time.time()
            else:
                start = time.time()
                output = model(batch)

                if isinstance(output["output_attention"], dict):
                    attention_pred = output["output_attention"]["final"]
                else:
                    attention_pred = output["output_attention"]

                attention_map = activation_func(attention_pred).cpu().detach().numpy()[0]

                # get the center of gravity from the attention map
                gaze_position = scipy.ndimage.center_of_mass(attention_map)
                gaze_position = [(gaze_position[1] / attention_map.shape[-1]) * 2.0 - 1.0,
                                 (gaze_position[0] / attention_map.shape[-2]) * 2.0 - 1.0]
                gaze_pos = gaze_position
                end = time.time()

            iterations += 1
            times.append(end - start)

            if iterations == config["iterations"]:
                break

    print("Results for model '{}' loaded from '{}'".format(model_config["model_name"], config["model_load_path"]))
    print("Total time over {} iterations is {}s".format(iterations, np.sum(times)))
    print("This results in a mean of {}s and a standard deviation of {}s".format(np.mean(times), np.std(times)))

    return temp, output, gaze_pos


def parse_config(args):
    config = vars(args)
    config["split_config"] = resolve_split_index_path(config["split_config"], config["data_root"])
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT_TEMP"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-m", "--model_load_path", type=str, default=None,
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-vn", "--video_name", type=str, default="flightmare_60",
                        help="The name of the input video.")
    parser.add_argument("-s", "--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Splits for which to create videos.")
    parser.add_argument("-sc", "--split_config", type=str, default=11,
                        help="TODO.")
    parser.add_argument("-it", "--iterations", type=int, default=5000,
                        help="Number of samples to iterate over.")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="The GPU to use.")
    parser.add_argument("-cg", "--compute_gaze", action="store_true",
                        help="Whether to factor computing gaze from an attention map into the equation. "
                             "Should only be specified when the model actually predicts attention maps.")

    # parse the arguments
    arguments = parse_config(parser.parse_args())

    # main
    measure_inference_time(arguments)
