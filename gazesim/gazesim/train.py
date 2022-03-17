import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from gazesim.training.config import parse_config
from gazesim.training.utils import get_batch_size, to_device, load_model
from gazesim.training.helpers import resolve_model_class, resolve_dataset_class, resolve_optimiser_class
from gazesim.training.helpers import resolve_losses, resolve_output_processing_func, resolve_logger_class
from gazesim.data.utils import pair


def train(config, device):
    # generators
    training_set = resolve_dataset_class(config["dataset_name"])(config, split="train")
    training_generator = DataLoader(training_set, batch_size=config["batch_size"],
                                    shuffle=True, num_workers=config["num_workers"])

    validation_set = resolve_dataset_class(config["dataset_name"])(config, split="val")
    validation_generator = DataLoader(validation_set, batch_size=config["batch_size"],
                                      shuffle=False, num_workers=config["num_workers"])

    # define the model
    model_class = resolve_model_class(config["model_name"])
    model = model_class(config)
    if config["model_info"] is not None:
        # TODO: need to update this to work with models where we want to partially load them
        #  => might be good to have a method for models like that which can be called
        model.load_state_dict(config["model_info"]["model_state_dict"])
    model = model.to(device)

    # define the optimiser
    optimiser = resolve_optimiser_class(config["optimiser"])(model.parameters(), lr=config["learning_rate"])
    if config["model_info"] is not None:
        optimiser.load_state_dict(config["model_info"]["optimiser_state_dict"])

    # define the loss function(s)
    loss_functions = resolve_losses(config["losses"])

    # define the logger
    logger = resolve_logger_class(config["dataset_name"], config["mode"])(config)
    logger.update_info(model=model, dataset=training_set)

    # prepare for doing pass over validation data args.validation_frequency times each epoch
    validation_check = np.linspace(0, len(training_set), config["validation_frequency"] + 1)
    validation_check = np.round(validation_check).astype(int)
    validation_check = validation_check[1:]

    # loop over epochs
    global_step = 0 if config["model_info"] is None else config["model_info"]["global_step"]
    for epoch in range(0 if config["model_info"] is None else config["model_info"]["epoch"] + 1, config["num_epochs"]):
        print("Starting epoch {:03d}!".format(epoch))
        model.train()
        validation_current = 0

        for batch_index, batch in tqdm(enumerate(training_generator), total=len(training_generator)):
            # transfer to GPU
            batch = to_device(batch, device)

            # forward pass, loss computation and backward pass
            optimiser.zero_grad()
            predictions = model(batch)
            total_loss = None
            partial_losses = {}
            for output in predictions:
                if isinstance(predictions[output], dict):
                    # this is very ugly, but for now it should work for the multi-scale attention model
                    partial_losses[output] = {}
                    for partial_output in predictions[output]:
                        current_prediction = resolve_output_processing_func(
                            output, config["losses"][output])(predictions[output][partial_output])
                        current_loss = loss_functions[output](current_prediction, batch[output])
                        if total_loss is None:
                            total_loss = current_loss
                        else:
                            total_loss += current_loss
                        partial_losses[output][partial_output] = current_loss
                else:
                    current_prediction = resolve_output_processing_func(
                        output, config["losses"][output])(predictions[output])
                    current_loss = loss_functions[output](current_prediction, batch[output])
                    if total_loss is None:
                        total_loss = current_loss
                    else:
                        total_loss += current_loss
                    partial_losses[output] = current_loss
            total_loss.backward()
            optimiser.step()

            with torch.no_grad():
                # global_step += batch[sorted(batch.keys())[0]].shape[0]
                global_step += get_batch_size(batch)

                # log at the end of each training step (each batch)
                # scalar_loss = loss.item()
                logger.training_step_end(global_step, total_loss, partial_losses, batch, predictions)

                # do validation if it should be done
                if (global_step - epoch * len(training_set)) >= validation_check[validation_current]:
                    disable = True
                    if config["validation_frequency"] == 1:
                        print("Validation for epoch {:03d}!".format(epoch))
                        disable = False

                    model.eval()
                    for val_batch_index, val_batch in tqdm(enumerate(validation_generator), disable=disable, total=len(validation_generator)):
                        # transfer to GPU
                        val_batch = to_device(val_batch, device)

                        # forward pass and loss computation
                        val_predictions = model(val_batch)
                        total_val_loss = None
                        partial_val_losses = {}
                        for output in val_predictions:
                            if isinstance(val_predictions[output], dict):
                                # this is very ugly, but for now it should work for the multi-scale attention model
                                partial_val_losses[output] = {}
                                for partial_output in val_predictions[output]:
                                    current_prediction = resolve_output_processing_func(
                                        output, config["losses"][output])(val_predictions[output][partial_output])
                                    current_loss = loss_functions[output](current_prediction, val_batch[output])
                                    if total_val_loss is None:
                                        total_val_loss = current_loss
                                    else:
                                        total_val_loss += current_loss
                                    partial_val_losses[output] = current_loss
                            else:
                                current_prediction = resolve_output_processing_func(
                                    output, config["losses"][output])(val_predictions[output])
                                current_loss = loss_functions[output](current_prediction, val_batch[output])
                                if total_val_loss is None:
                                    total_val_loss = current_loss
                                else:
                                    total_val_loss += current_loss
                                partial_val_losses[output] = current_loss

                        # tracking the loss in the logger
                        # val_scalar_loss = val_loss.item()
                        logger.validation_step_end(global_step, total_val_loss, partial_val_losses, val_batch, val_predictions)

                    # log after the complete pass over the validation set
                    logger.validation_epoch_end(global_step, epoch, model, optimiser)

                    # update index for checking whether we should run validation loop
                    validation_current += 1
                    model.train()

        # log at the end of the epoch
        logger.training_epoch_end(global_step, epoch, model, optimiser)


def cross_validate(config, device):
    # TODO: what should the CV logger log?
    #  - should the different CV runs be subdirectories or should they be stored as different scalars/variables?
    #    => maybe these should just be entirely different, e.g. "cv/loss/..."
    #  - do we even want to log e.g. training loss with cross validation?
    #    => probably should, just to see if everything's progressing
    #  - how should the final output of the cross-validation be saved?

    # define the classes here (could also be done in main...)
    dataset_class = resolve_dataset_class(config["dataset_name"])
    model_class = resolve_model_class(config["model_name"])
    optimiser_class = resolve_optimiser_class(config["optimiser"])

    # define the loss function(s)
    loss_functions = resolve_losses(config["losses"])

    # define the logger
    logger = resolve_logger_class(config["dataset_name"], config["mode"])(config)

    for cv_split in range(config["cv_splits"]):
        print("Starting split {:02d}!".format(cv_split))

        # generators
        training_set = dataset_class(config, split="train", cv_split=cv_split)
        training_generator = DataLoader(training_set, batch_size=config["batch_size"],
                                        shuffle=True, num_workers=config["num_workers"])

        validation_set = dataset_class(config, split="test", cv_split=cv_split)
        validation_generator = DataLoader(validation_set, batch_size=config["batch_size"],
                                          shuffle=False, num_workers=config["num_workers"])

        # define the model, no loading functionality here for now, would only make sense with something
        # like dreyeve and then it would probably take way to long to train for cross validation
        model = model_class(config)
        model = model.to(device)

        # update logger
        logger.update_info(model=model, dataset=training_set, split=cv_split)

        # define the optimiser, no loading same as for models
        optimiser = optimiser_class(model.parameters(), lr=config["learning_rate"])

        # prepare for doing pass over validation data args.validation_frequency times each epoch
        validation_check = np.linspace(0, len(training_set), config["validation_frequency"] + 1)
        validation_check = np.round(validation_check).astype(int)
        validation_check = validation_check[1:]

        # loop over epochs
        global_step = 0
        for epoch in range(config["num_epochs"]):
            print("Starting epoch {:03d}!".format(epoch))
            validation_current = 0

            model.train()
            for batch_index, batch in tqdm(enumerate(training_generator), total=len(training_generator)):
                # transfer to GPU
                batch = to_device(batch, device)

                # forward pass, loss computation and backward pass
                optimiser.zero_grad()
                predictions = model(batch)
                total_loss = None
                partial_losses = {}
                for output in predictions:
                    current_prediction = resolve_output_processing_func(output)(predictions[output])
                    current_loss = loss_functions[output](current_prediction, batch[output])
                    if total_loss is None:
                        total_loss = current_loss
                    else:
                        total_loss += current_loss
                    partial_losses[output] = total_loss
                total_loss.backward()
                optimiser.step()

                with torch.no_grad():
                    global_step += get_batch_size(batch)

                    # log at the end of each training step (each batch)
                    logger.training_step_end(global_step, total_loss, partial_losses, batch, predictions)

                    # do validation if it should be done
                    if (global_step - epoch * len(training_set)) >= validation_check[validation_current]:
                        disable = True
                        if config["validation_frequency"] == 1:
                            print("Validation for epoch {:03d}!".format(epoch))
                            disable = False

                        model.eval()
                        for val_batch_index, val_batch in tqdm(enumerate(validation_generator), disable=disable,
                                                               total=len(validation_generator)):
                            # transfer to GPU
                            val_batch = to_device(val_batch, device)

                            # forward pass and loss computation
                            val_predictions = model(val_batch)
                            total_val_loss = None
                            partial_val_losses = {}
                            for output in val_predictions:
                                current_prediction = resolve_output_processing_func(output)(val_predictions[output])
                                current_loss = loss_functions[output](current_prediction, val_batch[output])
                                if total_val_loss is None:
                                    total_val_loss = current_loss
                                else:
                                    total_val_loss += current_loss
                                partial_val_losses[output] = current_loss

                            # tracking the loss in the logger
                            # TODO: should probably also just record the validation error(s) for the
                            #  different splits in the logger and then save them in the logger as well
                            logger.validation_step_end(global_step, total_val_loss, partial_val_losses, val_batch,
                                                       val_predictions)

                        # log after the complete pass over the validation set
                        logger.validation_epoch_end(global_step, epoch, model, optimiser)

                        # update index for checking whether we should run validation loop
                        validation_current += 1
                        model.train()

            # log at the end of the epoch
            logger.training_epoch_end(global_step, epoch, model, optimiser)

            with torch.no_grad():
                model.eval()
                for train_batch_index, train_batch in tqdm(enumerate(training_generator), disable=disable,
                                                           total=len(training_generator)):
                    # transfer to GPU
                    train_batch = to_device(train_batch, device)

                    # forward pass and loss computation
                    train_predictions = model(train_batch)
                    total_train_loss = None
                    partial_train_losses = {}
                    for output in train_predictions:
                        current_prediction = resolve_output_processing_func(output)(train_predictions[output])
                        current_loss = loss_functions[output](current_prediction, train_batch[output])
                        if total_train_loss is None:
                            total_train_loss = current_loss
                        else:
                            total_train_loss += current_loss
                        partial_train_losses[output] = current_loss

                    # tracking the loss in the logger
                    logger.final_training_pass_step_end(global_step, total_train_loss, partial_train_losses,
                                                        train_batch, train_predictions)

                # log after the complete pass over the training set
                logger.final_training_pass_epoch_end(global_step, epoch, model, optimiser)
                model.train()


def val_or_test(config, device):
    # load the model
    model, model_config = load_model(config["model_load_path"], config["gpu"], return_config=True)
    model.to(device)
    model.eval()

    # modify model config for dataset loading
    model_config["data_root"] = config["data_root"]
    model_config["split_config"] = config["split_config"]
    model_config["eval_model_load_path"] = config["model_load_path"]

    # define the dataset
    validation_set = resolve_dataset_class(model_config["dataset_name"])(model_config, split=config["mode"])
    validation_generator = DataLoader(validation_set, batch_size=config["batch_size"],
                                      shuffle=False, num_workers=config["num_workers"])

    # define the loss function(s)
    loss_functions = resolve_losses(model_config["losses"])

    # define the logger and disable writing to hardware
    logger = resolve_logger_class(model_config["dataset_name"], config["mode"])(model_config, disable_write_to_disk=True)
    logger.update_info(model=model, dataset=validation_set)

    print("Starting validation!")
    with torch.no_grad():
        model.eval()
        for val_batch_index, val_batch in tqdm(enumerate(validation_generator), total=len(validation_generator)):
            # transfer to GPU
            val_batch = to_device(val_batch, device)

            # forward pass and loss computation
            val_predictions = model(val_batch)
            total_val_loss = None
            partial_val_losses = {}
            for output in val_predictions:
                if isinstance(val_predictions[output], dict):
                    # this is very ugly, but for now it should work for the multi-scale attention model
                    partial_val_losses[output] = {}
                    for partial_output in val_predictions[output]:
                        current_prediction = resolve_output_processing_func(
                            output, model_config["losses"][output])(val_predictions[output][partial_output])
                        current_loss = loss_functions[output](current_prediction, val_batch[output])
                        if total_val_loss is None:
                            total_val_loss = current_loss
                        else:
                            total_val_loss += current_loss
                        partial_val_losses[output] = current_loss
                else:
                    current_prediction = resolve_output_processing_func(
                        output, model_config["losses"][output])(val_predictions[output])
                    current_loss = loss_functions[output](current_prediction, val_batch[output])
                    if total_val_loss is None:
                        total_val_loss = current_loss
                    else:
                        total_val_loss += current_loss
                    partial_val_losses[output] = current_loss

            # tracking the loss in the logger
            logger.validation_step_end(0, total_val_loss, partial_val_losses, val_batch, val_predictions)

        # print out results after complete pass over validation set
        test = logger.total_loss_val / len(validation_generator)
        logger.validation_epoch_end(0, 0, model, None)

        print("Finished validation!")
        print("Validation loss :", test.item())


def main(config):
    # set the seed for PyTorch
    torch.manual_seed(config["torch_seed"])
    np.random.seed(config["torch_seed"])

    # use GPU if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(config["gpu"])
                          if use_cuda and config["gpu"] < torch.cuda.device_count() else "cpu")

    # check what to do: at the moment only choice between training and cross validation
    # would be easier to just specify that training/CV should be done rather than checking automatically...
    {"train": train, "cv": cross_validate, "val": val_or_test, "test": val_or_test}[config["mode"]](config, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # arguments related to the dataset
    parser.add_argument("-r", "--data_root", type=str,
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-sc", "--split_config",
                        help="The split configuration/index to get information about the division into training "
                             "and validation (and test) data from. Can either be the path to a file or an index "
                             "(will search in $DATA_ROOT/splits/).")
    parser.add_argument("-fps", "--frames_per_second", type=int,
                        help="Frame rate of the input videos (needs to be specified "
                             "since the original indexing is done at 60 FPS).")
    parser.add_argument("-ss", "--stack_size", type=int,
                        help="Number of frames to stack for models that take multiple "
                             "frames as input (i.e. C3D, Dr(eye)ve, RNNs)")
    parser.add_argument("-ivn", "--input_video_names", type=str, nargs="+",
                        help="The (file) name(s) for the video(s) to use as input.")
    parser.add_argument("-dsn", "--drone_state_names", type=str, nargs="+",
                        help="The column names/quantities to use as input when there is a drone state input. "
                             "Can also specify the following shorthands for pre-defined sets of columns: "
                             "'all', 'vel', 'acc', 'ang_vel'.")
    parser.add_argument("-agt", "--attention_ground_truth", type=str,
                        help="The (file) name(s) for the video(s) to use as targets for attention.")
    parser.add_argument("-cgt", "--control_ground_truth", type=str,
                        help="The (file) name(s) for the video(s) to use as targets for attention.")
    # parser.add_argument("-gtn", "--ground_truth_name", type=str,  # TODO: remove
    #                     help="The (file) name(s) for the video(s) to use as targets for attention.")
    parser.add_argument("-c", "--config_file", type=str,
                        help="Config file to load parameters from.")

    # arguments related to data normalisation/standardisation
    parser.add_argument("-nn", "--no_normalisation", action="store_true",
                        help="Whether or not to normalise the image input data.")
    parser.add_argument("-cn", "--control_normalisation", action="store_true",
                        help="Whether or not to normalise the control input data.")
    parser.add_argument("-cnr", "--control_normalisation_range", type=pair, nargs="+",
                        help="Ranges of control inputs to use for normalisation (i.e. maximum thrust and body rates).")
    parser.add_argument("-cg", "--clip_gaze", action="store_true",
                        help="Whether or not to clip gaze GT to [-1, 1] when loading it.")
    parser.add_argument("-sg", "--scale_gaze", action="store_true",
                        help="Whether or not to scale gaze GT to actual image coordinates (after clipping).")

    # arguments related to (video) data augmentation
    parser.add_argument("-vda", "--video_data_augmentation", action="store_true",
                        help="Whether or not to apply data augmentation for the image/video input data.")
    parser.add_argument("--vda_probability", type=float,
                        help="Probability of applying individual data augmentation transforms.")
    parser.add_argument("--vda_jitter_range", type=float,
                        help="Color jitter range for data augmentation.")
    parser.add_argument("--vda_gaussian_noise_sigma", type=float,
                        help="Standard deviation for Gaussian noise applied for data augmentation.")
    parser.add_argument("-vrc", "--video_random_cropping", action="store_true",
                        help="Whether or not to randomly crop image/video input data.")
    parser.add_argument("--vrc_factor_before_crop", type=float,
                        help="If random cropping is used, by what factor to "
                             "resize images/attention maps to before cropping.")

    # arguments related to the DDA input modalities
    parser.add_argument("-fts", "--feature_track_name", type=str,
                        help="The (file) name for already extracted feature tracks for the DDA architecture.")
    parser.add_argument("-ftn", "--feature_track_num", type=int,
                        help="The number of feature tracks per sample to use in the DDA architecture.")
    parser.add_argument("-rn", "--reference_name", type=str,
                        help="The (file) name for reference states for the DDA architecture.")
    parser.add_argument("-rv", "--reference_variables", type=str, nargs="+",
                        help="The column names/state variables to use for the reference statess for the DDA "
                             "architecture. Can also specify the following shorthands for pre-defined sets of "
                             "variables: 'all', 'pos', 'vel', 'acc', 'rot' 'omega'.")
    parser.add_argument("-sen", "--state_estimate_name", type=str,
                        help="The (file) name for state estimate measurements for the DDA architecture.")
    parser.add_argument("-sev", "--state_estimate_variables", type=str, nargs="+",
                        help="The column names/state variables to use for the state estimates for the DDA "
                             "architecture. Can also specify the following shorthands for pre-defined sets of "
                             "variables: 'all', 'pos', 'vel', 'acc', 'rot' 'omega'.")
    parser.add_argument("-seda", "--state_estimate_data_augmentation", action="store_true",
                        help="Whether or not to apply data augmentation to the provided state estimates for the"
                             "DDA architecture (in the form of adding Gaussian noise to the state variables).")

    # arguments related to the model
    parser.add_argument("-m", "--model_name", type=str,
                        choices=["codevilla", "c3d", "c3d_state", "codevilla300", "codevilla_skip",
                                 "codevilla_multi_head", "codevilla_dual_branch", "codevilla_no_state", "resnet_state",
                                 "resnet", "resnet_larger", "resnet_larger_dual_branch", "resnet_larger_multi_head",
                                 "resnet_state_larger", "resnet_larger_att_ctrl", "state_only", "dreyeve_branch",
                                 "resnet_att", "resnet_larger_gru", "ue4sim", "dda", "high_res_att", "simple_att",
                                 "resnet_gaze", "resnet_larger_gaze", "direct_supervision"],
                        help="The name of the model to use.")
    parser.add_argument("-mlp", "--model_load_path", type=str,  # TODO: maybe adjust for dreyeve net
                        help="Path to load a model checkpoint from (including information about the "
                             "architecture, the current weights and the state of the optimiser).")
    parser.add_argument("-nca", "--no_control_activation", action="store_true",
                        help="Whether or not to use an activation function for the output of the control "
                             "prediction network (currently using sigmoid for thrust and tanh for body rates).")
    parser.add_argument("-ga", "--gaze_activation", action="store_true",
                        help="Whether or not to use an activation function (tanh) "
                             "for the output of the gaze prediction networks.")
    parser.add_argument("-csf", "--channel_scale_factor", type=int,
                        help="Factor by which to  scale the number of channels for the high-resolution attention model.")
    parser.add_argument("-hra", "--high_res_activation", action="store_true",
                        help="Whether or not to activate the attention output of the "
                             "High-Res-Attention network (using hard tanh).")

    # arguments related to training
    parser.add_argument("-md", "--mode", type=str, choices=["train", "cv", "val", "test"],
                        help="Mode to train in, currently only 'normal' training and cross validation.")
    parser.add_argument("-g", "--gpu", type=int,
                        help="GPU to use for training if any are available.")
    parser.add_argument("-ts", "--torch_seed", type=int,
                        help="Random seed to use for calling torch.manual_seed(seed).")
    parser.add_argument("-w", "--num_workers", type=int,
                        help="Number of workers to use for loading the data.")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="Batch size to use for training.")
    parser.add_argument("-e", "--num_epochs", type=int,
                        help="Maximum number of epochs to train for.")
    parser.add_argument("-o", "--optimiser", type=str, choices=["adam"],
                        help="The optimiser to use.")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="The learning rate to start with.")
    parser.add_argument("-l", "--losses", type=str, nargs="+",
                        help="The loss to use. Depends on the model architecture and what kinds of outputs "
                             "(and how many) it has. For now only one loss can be specified (no architecture "
                             "with multiple outputs/losses). If the wrong loss is supplied, it will be changed "
                             "automatically to the default loss for a given architecture/output type.")

    # arguments related to logging information
    parser.add_argument("-lg", "--log_root", type=str,
                        help="Root directory where log folders for each run should be created.")
    parser.add_argument("-exp", "--experiment_name", type=str,
                        help="The name under which to save the logs and checkpoints (in addition to a timestamp).")
    parser.add_argument("-vf", "--validation_frequency", type=int,
                        help="How often to compute the validation loss during each epoch. When set to 1 "
                             "(the default value) this is only done at the end of the epoch, as is standard.")
    parser.add_argument("-cf", "--checkpoint_frequency", type=int,
                        help="Frequency at which to save model checkpoints (in epochs).")

    # parse the arguments
    arguments = parser.parse_args()

    # train
    main(parse_config(arguments))
