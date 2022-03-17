# TODO: load attention model, load respective dataset (by default validation split), evaluate
#  "normal loss" (KL-divergence), MSE (?), correlation coefficient on this data
#  => simultaneously do the same thing for at least the random gaze thingy
#     => simply create ToAttentionDataset?
#  => also load mean mask and compare the same thing
# I think this should be separate from gaze predictions or at least there should be 2 different functions?

import os
import numpy as np
import pandas as pd
import cv2
import torch
import scipy.ndimage
import time

from tqdm import tqdm
from scipy.stats import pearsonr
from contextlib import nullcontext
from gazesim.data.utils import resolve_split_index_path
from gazesim.data.datasets import ToAttentionDataset, ToGazeDataset
from gazesim.training.helpers import resolve_dataset_class
from gazesim.training.utils import to_device, to_batch, load_model
from gazesim.models.utils import image_softmax, image_log_softmax


def kld_numeric(y_true, y_pred):
    """
    Function to evaluate Kullback-Leiber divergence (sec 4.2.3 of [1]) on two samples.
    The two distributions are numpy arrays having arbitrary but coherent shapes.
    Taken from https://github.com/ndrplz/dreyeve/blob/5ba32174dff8fdbb5644b1cc8ecd2752308c06ce/experiments/metrics/metrics.py#L5
    :param y_true: groundtruth.
    :param y_pred: predictions.
    :return: numeric kld
    """
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    eps = np.finfo(np.float32).eps

    P = y_pred / (eps + np.sum(y_pred))  # prob
    Q = y_true / (eps + np.sum(y_true))  # prob

    kld = np.sum(Q * np.log(eps + Q / (eps + P)))

    return kld


def cc_numeric(y_pred, y_true):
    """
    Function to evaluate Pearson's correlation coefficient (sec 4.2.2 of [1]) on two samples.
    The two distributions are numpy arrays having arbitrary but coherent shapes.
    Taken from https://github.com/ndrplz/dreyeve/blob/5ba32174dff8fdbb5644b1cc8ecd2752308c06ce/experiments/metrics/metrics.py#L27
    :param y_true: groundtruth.
    :param y_pred: predictions.
    :return: numeric cc.
    """
    y_pred = y_pred.astype(np.float32)
    y_true = y_true.astype(np.float32)

    eps = np.finfo(np.float32).eps

    cv2.normalize(y_pred, dst=y_pred, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(y_true, dst=y_true, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    y_pred = y_pred.ravel()
    y_true = y_true.ravel()

    y_pred = (y_pred - np.mean(y_pred)) / (eps + np.std(y_pred))
    y_true = (y_true - np.mean(y_true)) / (eps + np.std(y_true))

    cc = np.corrcoef(y_pred, y_true)

    return cc[0][1]


def kl_divergence(prediction, target):
    prediction = prediction.squeeze().cpu().detach().numpy()
    target = target.squeeze().cpu().detach().numpy()
    kl_divs = []
    for idx in range(prediction.shape[0]):
        kl_divs.append(kld_numeric(prediction[idx], target[idx]))
    return np.mean(kl_divs)


def pearson_corr_coeff(prediction, target):
    prediction = prediction.reshape(prediction.shape[0], -1).cpu().detach().numpy()
    target = target.reshape(target.shape[0], -1).cpu().detach().numpy()
    coefficients = []
    for test in range(prediction.shape[0]):
        coefficients.append(pearsonr(prediction[test], target[test])[0])
    """
    prediction = prediction.squeeze().cpu().detach().numpy()
    target = target.squeeze().cpu().detach().numpy()
    coefficients = []
    for idx in range(prediction.shape[0]):
        coefficients.append(cc_numeric(prediction[idx], target[idx]))
    """
    return np.mean(coefficients)


def load(config):
    # use GPU if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(config["gpu"])
                          if use_cuda and config["gpu"] < torch.cuda.device_count() else "cpu")

    models, model_configs, model_names = {}, {}, []
    for mlp in config["model_load_path"]:
        print("Loading model from '{}'".format(mlp))
        start = time.time()

        model, model_config = load_model(mlp, config["gpu"], return_config=True)

        # get the model name (there could be duplicates in theory, would have to change if one wanted to test that)
        model_name = model_config["model_name"]
        model_names.append(model_name)

        # move model to correct device
        model.to(device)
        model.eval()
        models[model_name] = model

        # modify model config for dataset loading
        model_config["data_root"] = config["data_root"]
        model_config["split_config"] = config["split_config"]
        model_configs[model_name] = model_config

        # also store information about the load path
        model_config["eval_model_load_path"] = mlp

        print("Loaded model from '{}' after {}s".format(mlp, time.time() - start))

    return models, model_configs, model_names, device


def prepare_data(config, model_configs):
    datasets_gt, datasets_random_gt, mean_baselines = {}, {}, {}
    for model_name, model_config in model_configs.items():
        # need dataset to load original videos (for input) and ground-truth
        dataset_gt = resolve_dataset_class(model_config["dataset_name"])(model_config, split=config["split"], training=False)

        # also need to load random gaze attention GT
        random_gt_config = model_config.copy()
        if "gaze" in model_config["model_name"]:
            random_gt_config["gaze_ground_truth"] = "shuffled_random_frame_mean_gaze_gt"
            dataset_random_gt = ToGazeDataset(random_gt_config, split=config["split"], training=False)
        else:
            random_gt_config["attention_ground_truth"] = "shuffled_random_moving_window_frame_mean_gt"
            dataset_random_gt = ToAttentionDataset(random_gt_config, split=config["split"], training=False)

        # also load mean mask corresponding to the split
        if "gaze" in model_config["model_name"]:
            split_df = pd.read_csv(config["split_config"] + ".csv")
            gaze_df = pd.read_csv(os.path.join(config["data_root"], "index", "frame_mean_gaze_gt.csv"))
            gaze_df = gaze_df[~split_df["split"].isin(["none", config["split"]])]
            mean_baseline = gaze_df.mean().values
            if dataset_gt.output_scaling:
                mean_baseline *= np.array([800.0, 600.0])
            mean_baseline = {"output_gaze": mean_baseline}
        else:
            mean_baseline = cv2.imread(config["mean_mask_path"])
            mean_baseline = cv2.cvtColor(mean_baseline, cv2.COLOR_BGR2RGB)
            mean_baseline = dataset_gt.attention_output_transform(mean_baseline)
            mean_baseline = {"output_attention": mean_baseline}

        datasets_gt[model_name] = dataset_gt
        datasets_random_gt[model_name] = dataset_random_gt
        mean_baselines[model_name] = mean_baseline

    return datasets_gt, datasets_random_gt, mean_baselines


def evaluate_attention(config):
    # load model and config
    models, model_configs, model_names, device = load(config)

    # prepare the data(sets)
    datasets_gt, datasets_random_gaze, mean_masks = prepare_data(config, model_configs)

    # make sure they have the same number of frames
    assert len(set([len(ds) for ds in datasets_gt.values()] + [len(ds) for ds in datasets_random_gaze.values()])) <= 1, \
        "GT and random gaze dataset don't have the same length."
    dataset_length = len(list(datasets_gt.values())[0])

    # define loss functions
    loss_funcs = {
        "mse": torch.nn.MSELoss(),
        "kl": torch.nn.KLDivLoss(reduction="batchmean"),
        "cc": pearson_corr_coeff,
    }

    total_loss = {
        mn: {
            loss: {
                "prediction": 0,
                "random_gaze": 0,
                "mean_mask": 0,
            } for loss in loss_funcs
        } for mn in model_names
    }

    kl_not_inf = {
        mn: {
            "prediction": 0,
            "random_gaze": 0,
            "mean_mask": 0,
        } for mn in model_names
    }

    # loop through the dataset (without batching I guess? might be better to do it?)
    num_batches = 0
    for start_frame in tqdm(range(0, dataset_length, config["batch_size"])):
        # construct a batch (maybe just repeat mean map if we even use it)
        batch = {
            mn: {
                "prediction": [],
                "random_gaze": [],
                "mean_mask": []
            } for mn in model_names
        }
        for in_batch_idx in range(start_frame, start_frame + config["batch_size"]):
            if in_batch_idx >= dataset_length:
                break
            for mn, mb in batch.items():
                mb["prediction"].append(datasets_gt[mn][in_batch_idx])
                mb["random_gaze"].append(datasets_random_gaze[mn][in_batch_idx])
                mb["mean_mask"].append(mean_masks[mn])

        # transfer batches to GPU
        for mn, mb in batch.items():
            mb["prediction"] = to_device(to_batch(mb["prediction"]), device)
            mb["random_gaze"] = to_device(to_batch(mb["random_gaze"]), device)
            mb["mean_mask"] = to_device(to_batch(mb["mean_mask"]), device)

        # for each model, get the actual attention map (normalised etc.) and an "unactivated" version for KL-div
        attention_predictions = {}
        for mn in model_names:
            attention_predictions[mn] = {
                "gt": batch[mn]["prediction"]["output_attention"],
                "prediction": {},
                "random_gaze": {},
                "mean_mask": {},
            }

            attention_predictions[mn]["random_gaze"]["attention"] = batch[mn]["random_gaze"]["output_attention"]
            attention_predictions[mn]["random_gaze"]["log_attention"] = torch.log(
                attention_predictions[mn]["random_gaze"]["attention"])

            attention_predictions[mn]["mean_mask"]["attention"] = batch[mn]["mean_mask"]["output_attention"]
            attention_predictions[mn]["mean_mask"]["log_attention"] = torch.log(
                attention_predictions[mn]["mean_mask"]["attention"])

            # TODO: model predictions
            # first get the prediction
            output = models[mn](batch[mn]["prediction"])

            if isinstance(output["output_attention"], dict):
                attention_pred = output["output_attention"]["final"]
            else:
                attention_pred = output["output_attention"]

            # decide what to do with the prediction depending on the loss I guess
            if model_configs[mn]["losses"]["output_attention"] == "kl":
                # output is log_attention
                attention_predictions[mn]["prediction"]["log_attention"] = image_log_softmax(attention_pred)

                # need to convert to actual attention map for MSE/CC
                attention_predictions[mn]["prediction"]["attention"] = image_softmax(attention_pred)
            elif model_configs[mn]["losses"]["output_attention"] == "ice":
                """
                # maybe use this? but then the order stuff would be messed up
                output_processing_func = {
                    "kl": image_softmax,
                    "ice": torch.sigmoid,
                    "mse": lambda x: x,
                }
                """
                # output is pre-sigmoid logits
                attention_predictions[mn]["prediction"]["attention"] = torch.sigmoid(attention_pred)
                # TODO: might need to image_softmax this though...

                # need to first convert to log, softmax, then log again? or is just log-softmax enough
                # => question is whether softmax does anything different depending on whether the inputs
                #    are logits for sigmoid activation or actual probability values...
                attention_predictions[mn]["prediction"]["log_attention"] = image_log_softmax(attention_pred)
                # attention_predictions[mn]["prediction"]["log_attention"] = torch.log(attention_pred)
            elif model_configs[mn]["losses"]["output_attention"] == "mse":
                attention_predictions[mn]["prediction"]["attention"] = attention_pred

                # TODO: shouldn't this really just be the log? but that wouldn't necessarily give a valid distro right?
                attention_predictions[mn]["prediction"]["log_attention"] = image_log_softmax(attention_pred)
                # attention_predictions[mn]["prediction"]["log_attention"] = torch.log(attention_pred)

        # compute the losses between all the models and the ground-truth
        for mn, pred in attention_predictions.items():
            # also need to loop over the prediction and so on...
            for mod, mod_pred in pred.items():
                if mod != "gt":
                    for loss, loss_f in loss_funcs.items():
                        # only use log_attention for KL
                        if loss == "kl":
                            current_loss = loss_f(mod_pred["log_attention"], pred["gt"]).item()

                            # only count if this is non-inf
                            if not np.isinf(current_loss):
                                total_loss[mn][loss][mod] += current_loss
                                kl_not_inf[mn][mod] += 1
                        else:
                            current_loss = loss_f(mod_pred["attention"], pred["gt"]).item()
                            total_loss[mn][loss][mod] += current_loss

        num_batches += 1

    with (nullcontext() if config["output_file"] is None else open(config["output_file"], "w")) as f:
        print("\n-----------------------------------------------------------------------------------------------\n")
        if config["output_file"] is not None:
            print("\n-----------------------------------------------------"
                  "------------------------------------------\n", file=f)
        for mn, model_res in total_loss.items():
            print("Model load path: {}\n".format(model_configs[mn]["eval_model_load_path"]))
            if config["output_file"] is not None:
                print("Model load path: {}\n".format(model_configs[mn]["eval_model_load_path"]), file=f)
            for loss, loss_res in model_res.items():
                for mod, mod_res in loss_res.items():
                    if loss == "kl":
                        if kl_not_inf[mn][mod] > 0:
                            average_loss = mod_res / kl_not_inf[mn][mod]
                        else:
                            print("Average {} loss for model {} comparing {} and GT is: infinite".format(
                                loss.upper(), mn, mod))
                            if config["output_file"] is not None:
                                print("Average {} loss for model {} comparing {} and GT is: infinite".format(
                                    loss.upper(), mn, mod), file=f)
                            continue
                    else:
                        average_loss = mod_res / num_batches

                    print("Average {} loss for model {} comparing {} and GT is: {}".format(
                        loss.upper(), mn, mod, average_loss))
                    if config["output_file"] is not None:
                        print("Average {} loss for model {} comparing {} and GT is: {}".format(
                            loss.upper(), mn, mod, average_loss), file=f)
                print()
                if config["output_file"] is not None:
                    print(file=f)
            print("-----------------------------------------------------------------------------------------------\n")
            if config["output_file"] is not None:
                print("-----------------------------------------------------"
                      "------------------------------------------\n", file=f)


def evaluate_gaze(config):
    # load model and config
    models, model_configs, model_names, device = load(config)

    # prepare the data(sets)
    datasets_gt, datasets_random_gaze, mean_masks = prepare_data(config, model_configs)

    # make sure they have the same number of frames
    assert len(set([len(ds) for ds in datasets_gt.values()] + [len(ds) for ds in datasets_random_gaze.values()])) <= 1, \
        "GT and random gaze dataset don't have the same length."
    dataset_length = len(list(datasets_gt.values())[0])

    # define loss functions
    loss_funcs = {
        "mse": torch.nn.MSELoss(),
        "l1": torch.nn.L1Loss(),
    }

    total_loss = {
        mn: {
            loss: {
                "prediction": {
                    "total": 0,
                    "partial_x": 0,
                    "partial_y": 0,
                },
                "random_gaze": {
                    "total": 0,
                    "partial_x": 0,
                    "partial_y": 0,
                },
                "mean_mask": {
                    "total": 0,
                    "partial_x": 0,
                    "partial_y": 0,
                },
            } if "gaze" in mn else {
                "prediction": {
                    "total": 0,
                    "partial_x": 0,
                    "partial_y": 0,
                },
            } for loss in loss_funcs
        } for mn in model_names
    }

    # loop through the dataset (without batching I guess? might be better to do it?)
    num_batches = 0
    for start_frame in tqdm(range(0, dataset_length, config["batch_size"])):
        # construct a batch (maybe just repeat mean map if we even use it)
        batch = {
            mn: {
                "prediction": [],
                "random_gaze": [],
                "mean_mask": []
            } if "gaze" in mn else {"prediction": []} for mn in model_names
        }
        for in_batch_idx in range(start_frame, start_frame + config["batch_size"]):
            if in_batch_idx >= dataset_length:
                break
            for mn, mb in batch.items():
                mb["prediction"].append(datasets_gt[mn][in_batch_idx])
                if "gaze" in mn:
                    mb["random_gaze"].append(datasets_random_gaze[mn][in_batch_idx])
                    mb["mean_mask"].append(mean_masks[mn])

        # transfer batches to GPU
        for mn, mb in batch.items():
            mb["prediction"] = to_device(to_batch(mb["prediction"]), device)
            if "gaze" in mn:
                mb["random_gaze"] = to_device(to_batch(mb["random_gaze"]), device)
                mb["mean_mask"] = to_device(to_batch(mb["mean_mask"]), device)

        ground_truth_batch = None
        for mn, mb in batch.items():
            if "gaze" in mn:
                ground_truth_batch = mb["prediction"]["output_gaze"]  # I think??
                if datasets_gt[mn].output_scaling:
                    mb["prediction"]["output_gaze"] /= torch.tensor(
                        [800.0, 600.0], dtype=mb["prediction"]["output_gaze"].dtype, device=device)

        # if the model is not a gaze model, get the attention map and compute the center of mass etc.
        gaze_predictions = {}
        for mn in model_names:
            # TODO: determine whether output is gaze or attention

            # only need to save the actual prediction here
            """
            gaze_predictions[mn] = {
                # "gt": batch[mn]["prediction"]["output_gaze"],  # TODO: this need to be just from the actual gaze thing!!!!!!!!!
                # => maybe just put this in one thing outside of this loop
                # => also needs to be normalised (potentially)
                "prediction": None,
                "random_gaze": None,
                "mean_mask": None,
            }
            """
            gaze_predictions[mn] = {}

            # gaze_predictions[mn]["random_gaze"] = batch[mn]["random_gaze"]["output_attention"]
            # gaze_predictions[mn]["mean_mask"] = batch[mn]["mean_mask"]["output_attention"]
            # ok yeah nvm

            # first get the prediction
            output = models[mn](batch[mn]["prediction"])

            # check if model is gaze model
            if "gaze" in mn:
                # get the prediction
                gaze_predictions[mn]["prediction"] = output["output_gaze"]
                gaze_predictions[mn]["random_gaze"] = batch[mn]["random_gaze"]["output_gaze"]
                gaze_predictions[mn]["mean_mask"] = batch[mn]["mean_mask"]["output_gaze"]
                if datasets_gt[mn].output_scaling:
                    gaze_predictions[mn]["prediction"] /= torch.tensor(
                        [800.0, 600.0], dtype=gaze_predictions[mn]["prediction"].dtype, device=device)
                    gaze_predictions[mn]["random_gaze"] /= torch.tensor(
                        [800.0, 600.0], dtype=gaze_predictions[mn]["random_gaze"].dtype, device=device)
                    gaze_predictions[mn]["mean_mask"] /= torch.tensor(
                        [800.0, 600.0], dtype=gaze_predictions[mn]["mean_mask"].dtype, device=device)
            else:
                # this of course assumes that only gaze or attention
                # models are passed as input, anything else is user error
                if isinstance(output["output_attention"], dict):
                    attention_pred = output["output_attention"]["final"]
                else:
                    attention_pred = output["output_attention"]

                attention_map = None
                if model_configs[mn]["losses"]["output_attention"] == "kl":
                    attention_map = image_softmax(attention_pred)
                elif model_configs[mn]["losses"]["output_attention"] == "ice":
                    attention_map = torch.sigmoid(attention_pred)
                elif model_configs[mn]["losses"]["output_attention"] == "mse":
                    attention_map = attention_pred

                # get the center of gravity from the attention map
                gaze_positions = []
                attention_map = attention_map.cpu().detach().numpy()
                for sample in attention_map:
                    gaze_position = scipy.ndimage.center_of_mass(sample.squeeze())
                    gaze_position = [(gaze_position[1] / attention_map.shape[-1]) * 2.0 - 1.0,
                                     (gaze_position[0] / attention_map.shape[-2]) * 2.0 - 1.0]
                    gaze_position = torch.tensor(gaze_position)
                    gaze_positions.append(gaze_position)
                gaze_positions = torch.stack(gaze_positions, dim=0).to(device)

                gaze_predictions[mn]["prediction"] = gaze_positions

                """
                print("predictions by attention model:", gaze_predictions[mn]["prediction"])

                # should probably check with image
                test = attention_map.squeeze()[0]
                test = test / test.max()
                test = (test * 255.0).astype(np.uint8)
                test = np.repeat(test[np.newaxis, :, :], 3, axis=0).transpose((1, 2, 0))
                test = cv2.cvtColor(cv2.cvtColor(test, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)
                print(test.shape, test.dtype)

                center = gaze_positions[0].cpu().numpy()
                print("center before:", center)
                print("what x", (center[0] + 1) / 2 * test.shape[1], test.shape[1])
                print("what y", (center[1] + 1) / 2 * test.shape[0], test.shape[0])
                center = (int((center[0] + 1) / 2 * test.shape[1]), int((center[1] + 1) / 2 * test.shape[0]))
                print("center:", center)
                test = cv2.circle(test, center, 5, (0, 0, 255), -1)

                other_center = ground_truth_batch[0].cpu().numpy()
                other_center = (int((other_center[0] + 1) / 2 * test.shape[1]), int((other_center[1] + 1) / 2 * test.shape[0]))
                test = cv2.circle(test, other_center, 5, (255, 0, 0), -1)

                cv2.imshow("", test)
                cv2.waitKey(0)
                """

        # compute the losses between all the models and the ground-truth
        for mn, pred in gaze_predictions.items():
            # also need to loop over the prediction and so on...
            for mod, mod_pred in pred.items():
                # if "prediction" or "gaze" in mn:
                for loss, loss_f in loss_funcs.items():
                    current_total_loss = loss_f(mod_pred, ground_truth_batch).item()
                    current_partial_x_loss = loss_f(mod_pred[:, 0], ground_truth_batch[:, 0]).item()
                    current_partial_y_loss = loss_f(mod_pred[:, 1], ground_truth_batch[:, 1]).item()
                    total_loss[mn][loss][mod]["total"] += current_total_loss
                    total_loss[mn][loss][mod]["partial_x"] += current_partial_x_loss
                    total_loss[mn][loss][mod]["partial_y"] += current_partial_y_loss

        num_batches += 1

        # if num_batches > 100:
        #     break

    with (nullcontext if config["output_file"] is None else open(config["output_file"], "w")) as f:
        print("\n-----------------------------------------------------------------------------------------------\n")
        if config["output_file"] is not None:
            print("\n-----------------------------------------------------"
                  "------------------------------------------\n", file=f)
        for mn, model_res in total_loss.items():
            print("Model load path: {}\n".format(model_configs[mn]["eval_model_load_path"]))
            if config["output_file"] is not None:
                print("Model load path: {}\n".format(model_configs[mn]["eval_model_load_path"]), file=f)
            for loss, loss_res in model_res.items():
                for mod, mod_res in loss_res.items():
                    for pn, partial in mod_res.items():
                        average_loss = partial / num_batches
                        print("Average {} loss for model {} comparing {} ({}) and GT is: {}".format(
                            loss.upper(), mn, mod, pn, average_loss))
                        if config["output_file"] is not None:
                            print("Average {} loss for model {} comparing {} ({}) and GT is: {}".format(
                                loss.upper(), mn, mod, pn, average_loss), file=f)
                    print()
                    if config["output_file"] is not None:
                        print(file=f)
                print()
                if config["output_file"] is not None:
                    print(file=f)
            print("-----------------------------------------------------------------------------------------------\n")
            if config["output_file"] is not None:
                print("-----------------------------------------------------"
                      "------------------------------------------\n", file=f)


def old_evaluate_gaze(config):
    # load model and config
    model, model_config, device = load(config)

    # prepare the data
    dataset_gt, dataset_random_gaze, mean_gaze = prepare_data(config, model_config)

    # loss functions
    loss_func_mse = torch.nn.MSELoss()
    loss_func_l1 = torch.nn.L1Loss()

    total_loss_pred_mse = 0
    total_loss_pred_mse_partial_x = 0
    total_loss_pred_mse_partial_y = 0
    total_loss_pred_l1 = 0
    total_loss_pred_l1_partial_x = 0
    total_loss_pred_l1_partial_y = 0

    total_loss_random_gaze_mse = 0
    total_loss_random_gaze_mse_partial_x = 0
    total_loss_random_gaze_mse_partial_y = 0
    total_loss_random_gaze_l1 = 0
    total_loss_random_gaze_l1_partial_x = 0
    total_loss_random_gaze_l1_partial_y = 0

    total_loss_mean_gaze_mse = 0
    total_loss_mean_gaze_mse_partial_x = 0
    total_loss_mean_gaze_mse_partial_y = 0
    total_loss_mean_gaze_l1 = 0
    total_loss_mean_gaze_l1_partial_x = 0
    total_loss_mean_gaze_l1_partial_y = 0

    # loop through the dataset (without batching I guess? might be better to do it?)
    num_batches = 0
    for start_frame in tqdm(range(0, len(dataset_gt), config["batch_size"])):
        # construct a batch (maybe just repeat mean map if we even use it)
        batch_gt = []
        batch_random_gaze = []
        batch_mean_gaze = []
        for in_batch_idx in range(start_frame, start_frame + config["batch_size"]):
            if in_batch_idx >= len(dataset_gt):
                break
            batch_gt.append(dataset_gt[in_batch_idx])
            batch_random_gaze.append(dataset_random_gaze[in_batch_idx])
            batch_mean_gaze.append(mean_gaze)
        batch_gt = to_device(to_batch(batch_gt), device)
        batch_random_gaze = to_device(to_batch(batch_random_gaze), device)
        batch_mean_gaze = to_device(to_batch(batch_mean_gaze), device)

        # get the attention ground-truth from the batch
        gaze_gt = batch_gt["output_gaze"]
        gaze_random_gaze = batch_random_gaze["output_gaze"]
        gaze_mean_gaze = batch_mean_gaze["output_gaze"]
        if dataset_gt.output_scaling:
            gaze_gt /= torch.tensor([800.0, 600.0], dtype=gaze_gt.dtype, device=device)
            gaze_random_gaze /= torch.tensor([800.0, 600.0], dtype=gaze_random_gaze.dtype, device=device)
            gaze_mean_gaze /= torch.tensor([800.0, 600.0], dtype=gaze_mean_gaze.dtype, device=device)

        # compute the attention prediction
        output = model(batch_gt)
        gaze_pred = output["output_gaze"]
        if dataset_gt.output_scaling:
            gaze_pred /= torch.tensor([800.0, 600.0], dtype=gaze_pred.dtype, device=device)

        # compute the losses between the "actual" ground-truth and the other stuff
        loss_pred_mse = loss_func_mse(gaze_pred, gaze_gt).item()
        loss_mse_pred_partial_x = loss_func_mse(gaze_pred[:, 0], gaze_gt[:, 0]).item()
        loss_mse_pred_partial_y = loss_func_mse(gaze_pred[:, 1], gaze_gt[:, 1]).item()
        loss_pred_l1 = loss_func_l1(gaze_pred, gaze_gt).item()
        loss_l1_pred_partial_x = loss_func_l1(gaze_pred[:, 0], gaze_gt[:, 0]).item()
        loss_l1_pred_partial_y = loss_func_l1(gaze_pred[:, 1], gaze_gt[:, 1]).item()

        loss_random_gaze_mse = loss_func_mse(gaze_random_gaze, gaze_gt).item()
        loss_mse_random_gaze_partial_x = loss_func_mse(gaze_random_gaze[:, 0], gaze_gt[:, 0]).item()
        loss_mse_random_gaze_partial_y = loss_func_mse(gaze_random_gaze[:, 1], gaze_gt[:, 1]).item()
        loss_random_gaze_l1 = loss_func_l1(gaze_random_gaze, gaze_gt).item()
        loss_l1_random_gaze_partial_x = loss_func_l1(gaze_random_gaze[:, 0], gaze_gt[:, 0]).item()
        loss_l1_random_gaze_partial_y = loss_func_l1(gaze_random_gaze[:, 1], gaze_gt[:, 1]).item()

        loss_mean_gaze_mse = loss_func_mse(gaze_mean_gaze, gaze_gt).item()
        loss_mse_mean_gaze_partial_x = loss_func_mse(gaze_mean_gaze[:, 0], gaze_gt[:, 0]).item()
        loss_mse_mean_gaze_partial_y = loss_func_mse(gaze_mean_gaze[:, 1], gaze_gt[:, 1]).item()
        loss_mean_gaze_l1 = loss_func_l1(gaze_mean_gaze, gaze_gt).item()
        loss_l1_mean_gaze_partial_x = loss_func_l1(gaze_mean_gaze[:, 0], gaze_gt[:, 0]).item()
        loss_l1_mean_gaze_partial_y = loss_func_l1(gaze_mean_gaze[:, 1], gaze_gt[:, 1]).item()

        # sum losses
        total_loss_pred_mse += loss_pred_mse
        total_loss_pred_mse_partial_x += loss_mse_pred_partial_x
        total_loss_pred_mse_partial_y += loss_mse_pred_partial_y
        total_loss_pred_l1 += loss_pred_l1
        total_loss_pred_l1_partial_x += loss_l1_pred_partial_x
        total_loss_pred_l1_partial_y += loss_l1_pred_partial_y

        total_loss_random_gaze_mse += loss_random_gaze_mse
        total_loss_random_gaze_mse_partial_x += loss_mse_random_gaze_partial_x
        total_loss_random_gaze_mse_partial_y += loss_mse_random_gaze_partial_y
        total_loss_random_gaze_l1 += loss_random_gaze_l1
        total_loss_random_gaze_l1_partial_x += loss_l1_random_gaze_partial_x
        total_loss_random_gaze_l1_partial_y += loss_l1_random_gaze_partial_y

        total_loss_mean_gaze_mse += loss_mean_gaze_mse
        total_loss_mean_gaze_mse_partial_x += loss_mse_mean_gaze_partial_x
        total_loss_mean_gaze_mse_partial_y += loss_mse_mean_gaze_partial_y
        total_loss_mean_gaze_l1 += loss_mean_gaze_l1
        total_loss_mean_gaze_l1_partial_x += loss_l1_mean_gaze_partial_x
        total_loss_mean_gaze_l1_partial_y += loss_l1_mean_gaze_partial_y

        num_batches += 1

    print("Average MSE loss for gaze model: {}".format(total_loss_pred_mse / num_batches))
    print("Average partial MSE loss (x-axis) for gaze model: {}".format(total_loss_pred_mse_partial_x / num_batches))
    print("Average partial MSE loss (y-axis) for gaze model: {}".format(total_loss_pred_mse_partial_y / num_batches))

    print("\nAverage L1 loss for gaze model: {}".format(total_loss_pred_l1 / num_batches))
    print("Average partial L1 loss (x-axis) for gaze model: {}".format(total_loss_pred_l1_partial_x / num_batches))
    print("Average partial L1 loss (y-axis) for gaze model: {}".format(total_loss_pred_l1_partial_y / num_batches))

    print("\n-------------------------------------------------------------------")

    print("\nAverage MSE loss for random gaze: {}".format(total_loss_random_gaze_mse / num_batches))
    print("Average partial MSE loss (x-axis) for random gaze: {}".format(total_loss_random_gaze_mse_partial_x / num_batches))
    print("Average partial MSE loss (y-axis) for random gaze: {}".format(total_loss_random_gaze_mse_partial_y / num_batches))

    print("\nAverage L1 loss for random gaze: {}".format(total_loss_random_gaze_l1 / num_batches))
    print("Average partial L1 loss (x-axis) for random gaze: {}".format(total_loss_random_gaze_l1_partial_x / num_batches))
    print("Average partial L1 loss (y-axis) for random gaze: {}".format(total_loss_random_gaze_l1_partial_y / num_batches))

    print("\n-------------------------------------------------------------------")

    print("\nAverage MSE loss for mean gaze: {}".format(total_loss_mean_gaze_mse / num_batches))
    print("Average partial MSE loss (x-axis) for mean gaze: {}".format(total_loss_mean_gaze_mse_partial_x / num_batches))
    print("Average partial MSE loss (y-axis) for mean gaze: {}".format(total_loss_mean_gaze_mse_partial_y / num_batches))

    print("\nAverage L1 loss for mean gaze: {}".format(total_loss_mean_gaze_l1 / num_batches))
    print("Average partial L1 loss (x-axis) for mean gaze: {}".format(total_loss_mean_gaze_l1_partial_x / num_batches))
    print("Average partial L1 loss (y-axis) for mean gaze: {}".format(total_loss_mean_gaze_l1_partial_y / num_batches))


def parse_config(args):
    config = vars(args)
    config["split_config"] = resolve_split_index_path(config["split_config"], config["data_root"])
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-m", "--model_load_path", type=str, nargs="+", default=None,
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-mm", "--mean_mask_path", type=str,
                        default=os.path.join(os.getenv("GAZESIM_ROOT"), "preprocessing_info", "split011_mean_mask.png"),
                        help="The path to the mean mask to compare stuff against.")
    parser.add_argument("-of", "--output_file", type=str, default=None,
                        help="File to save the final metrics to.")
    parser.add_argument("-md", "--mode", type=str, default="attention", choices=["attention", "gaze"],
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-vn", "--video_name", type=str, default="flightmare_60",
                        help="The name of the input video.")
    parser.add_argument("-s", "--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Splits for which to create videos.")
    parser.add_argument("-sc", "--split_config", type=str, default=11,
                        help="TODO.")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="The GPU to use.")
    parser.add_argument("-b", "--batch_size", type=int, default=1,
                        help="Batch size.")

    # parse the arguments
    arguments = parse_config(parser.parse_args())

    # main
    if arguments["mode"] == "attention":
        evaluate_attention(arguments)
    elif arguments["mode"] == "gaze":
        evaluate_gaze(arguments)
