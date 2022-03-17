import os
import json
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from gazesim.models.utils import image_softmax, convert_attention_to_image


def save_config(config, config_save_path):
    # remove those entries that cannot be saved with json
    excluded = ["model_info"]
    config_to_save = {}
    for k in config:
        if k not in excluded:
            config_to_save[k] = config[k]

    # save the config
    with open(config_save_path, "w") as f:
        json.dump(config_to_save, f)


class Logger:

    def __init__(self, config, disable_write_to_disk=False):
        self.disable_write_to_disk = disable_write_to_disk

        # determine how many splits there are
        if config["mode"] == "cv":
            self.splits = config["cv_splits"]
        else:
            self.splits = 1
        self.current_split = 0

        # TODO: maybe define some prefixes and stuff (e.g. should cv logs start with "cv/..."?
        #  although things should work as they are now...

        # create log and checkpoint directories
        self.log_dir = os.path.join(config["log_root"], config["experiment_name"])
        self.tensorboard_dirs = []
        self.checkpoint_dirs = []
        if not self.disable_write_to_disk and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not self.disable_write_to_disk:
            if self.splits == 1:
                self.tensorboard_dirs.append(os.path.join(self.log_dir, "tensorboard"))
                self.checkpoint_dirs.append(os.path.join(self.log_dir, "checkpoints"))
                if not os.path.exists(self.tensorboard_dirs[-1]):
                    os.makedirs(self.tensorboard_dirs[-1])
                if not os.path.exists(self.checkpoint_dirs[-1]):
                    os.makedirs(self.checkpoint_dirs[-1])
            else:
                for i in range(self.splits):
                    self.tensorboard_dirs.append(os.path.join(self.log_dir, "tensorboard", f"split_{i}"))
                    self.checkpoint_dirs.append(os.path.join(self.log_dir, "checkpoints", f"split_{i}"))
                    if not os.path.exists(self.tensorboard_dirs[-1]):
                        os.makedirs(self.tensorboard_dirs[-1])
                    if not os.path.exists(self.checkpoint_dirs[-1]):
                        os.makedirs(self.checkpoint_dirs[-1])

        # create tensorboard writers
        self.tb_writers = []
        if not self.disable_write_to_disk:
            for tb_dir in self.tensorboard_dirs:
                self.tb_writers.append(SummaryWriter(tb_dir))

        # store config for information and save config file
        self.config = config
        if not self.disable_write_to_disk:
            save_config(self.config, os.path.join(self.log_dir, "config.json"))

    def update_info(self, **kwargs):
        pass

    def training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # to be called when a single training step (1 batch) is performed
        raise NotImplementedError()

    def training_epoch_end(self, global_step, epoch, model, optimiser):
        # to be called after each full pass over the training set
        if self.disable_write_to_disk:
            print("Epoch {:03d}: Saving checkpoint is disabled!".format(epoch))
            return

        # save model checkpoint
        if (epoch + 1) % self.config["checkpoint_frequency"] == 0:
            torch.save({
                "global_step": global_step,
                "epoch": epoch,
                "model_name": self.config["model_name"],
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict()
            }, os.path.join(self.checkpoint_dirs[self.current_split], "epoch{:03d}.pt".format(epoch)))
            print("Epoch {:03d}: Saving checkpoint to '{}'".format(epoch, self.checkpoint_dirs[self.current_split]))

    def validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # to be called when a single validation step (1 batch) is performed
        raise NotImplementedError()

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        # to be called after each full pass over the validation set
        raise NotImplementedError()

    def final_training_pass_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        raise NotImplementedError()

    def final_training_pass_epoch_end(self, global_step, epoch, model, optimiser):
        raise NotImplementedError()


class TestLogger(Logger):

    def training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        pass

    def training_epoch_end(self, global_step, epoch, model, optimiser):
        pass

    def validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        pass

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        pass

    def final_training_pass_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        pass

    def final_training_pass_epoch_end(self, global_step, epoch, model, optimiser):
        pass


class GenericLogger(Logger):

    def __init__(self, config, disable_write_to_disk=False):
        super().__init__(config, disable_write_to_disk)

        self.total_loss_val = None
        self.counter_val = 0

    def _generic_training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # log total loss
        if not self.disable_write_to_disk:
            self.tb_writers[self.current_split].add_scalar("loss/train/total", total_loss.item(), global_step)

    def _generic_validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # accumulate total loss
        if self.total_loss_val is None:
            self.total_loss_val = torch.zeros_like(total_loss)
        self.total_loss_val += total_loss

    def _generic_validation_epoch_end(self, global_step, epoch, model, optimiser):
        # log total loss
        if not self.disable_write_to_disk:
            self.tb_writers[self.current_split].add_scalar(
                "loss/val/total", self.total_loss_val.item() / self.counter_val, global_step)

        # reset the loss accumulator
        self.total_loss_val = torch.zeros_like(self.total_loss_val)

    def training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        self._generic_training_step_end(global_step, total_loss, partial_losses, batch, predictions)

    def validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        self._generic_validation_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self.counter_val += 1

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        self._generic_validation_epoch_end(global_step, epoch, model, optimiser)
        self.counter_val = 0

    def final_training_pass_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        raise NotImplementedError

    def final_training_pass_epoch_end(self, global_step, epoch, model, optimiser):
        raise NotImplementedError


class ControlLogger(GenericLogger):

    def __init__(self, config, disable_write_to_disk=False):
        super().__init__(config, disable_write_to_disk)

        # should these be lists? I don't think they need to be, since we usually only need to keep track of one
        # can probably just subclass this class and have a current_split index....
        self.control_names = None
        self.control_partial_losses_val_mse = None
        self.control_partial_losses_val_l1 = None

        self.control_total_loss_val_l1 = None

    def update_info(self, **kwargs):
        if "dataset" in kwargs:
            self.control_names = kwargs.get("dataset").output_columns

    def _control_training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # determine individual losses
        individual_losses_mse = F.mse_loss(predictions["output_control"],
                                           batch["output_control"],
                                           reduction="none")
        individual_losses_mse = torch.mean(individual_losses_mse, dim=0)

        # log individual losses
        if not self.disable_write_to_disk:
            for n, l_mse in zip(self.control_names, individual_losses_mse):
                self.tb_writers[self.current_split].add_scalar(f"loss/train/output_control/{n}/mse", l_mse, global_step)

    def _control_validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # determine individual losses
        individual_losses_mse = F.mse_loss(predictions["output_control"],
                                           batch["output_control"],
                                           reduction="none")
        individual_losses_mse = torch.mean(individual_losses_mse, dim=0)
        individual_losses_l1 = F.l1_loss(predictions["output_control"],
                                         batch["output_control"],
                                         reduction="none")
        individual_losses_l1 = torch.mean(individual_losses_l1, dim=0)

        # TODO: maybe add back total l1 loss as well?
        if self.control_total_loss_val_l1 is None:
            self.control_total_loss_val_l1 = torch.zeros_like(total_loss)
        self.control_total_loss_val_l1 += torch.mean(individual_losses_l1)

        # accumulate individual losses
        if self.control_partial_losses_val_mse is None:
            self.control_partial_losses_val_mse = torch.zeros_like(individual_losses_mse)
            self.control_partial_losses_val_l1 = torch.zeros_like(individual_losses_l1)
        self.control_partial_losses_val_mse += individual_losses_mse
        self.control_partial_losses_val_l1 += individual_losses_l1

    def _control_validation_epoch_end(self, global_step, epoch, model, optimiser):
        # log individual losses
        if not self.disable_write_to_disk:
            for n, l_mse, l_l1 in zip(self.control_names, self.control_partial_losses_val_mse,
                                      self.control_partial_losses_val_l1):
                self.tb_writers[self.current_split].add_scalar(f"loss/val/output_control/{n}/mse",
                                                               l_mse / self.counter_val, global_step)
                self.tb_writers[self.current_split].add_scalar(f"loss/val/output_control/{n}/l1",
                                                               l_l1 / self.counter_val, global_step)

        # reset the loss accumulators
        self.control_partial_losses_val_mse = torch.zeros_like(self.control_partial_losses_val_mse)
        self.control_partial_losses_val_l1 = torch.zeros_like(self.control_partial_losses_val_l1)

    def training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        self._generic_training_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self._control_training_step_end(global_step, total_loss, partial_losses, batch, predictions)

    def validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        self._generic_validation_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self._control_validation_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self.counter_val += 1

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        self._generic_validation_epoch_end(global_step, epoch, model, optimiser)
        total_loss_val_l1 = self.control_total_loss_val_l1.item() / self.counter_val
        self._control_validation_epoch_end(global_step, epoch, model, optimiser)
        self.counter_val = 0
        return total_loss_val_l1

    def final_training_pass_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        raise NotImplementedError()

    def final_training_pass_epoch_end(self, global_step, epoch, model, optimiser):
        raise NotImplementedError()


class GazeLogger(GenericLogger):

    def __init__(self, config, disable_write_to_disk=False):
        super().__init__(config, disable_write_to_disk)

        # should these be lists? I don't think they need to be, since we usually only need to keep track of one
        # can probably just subclass this class and have a current_split index....
        self.gaze_names = ["x_norm_coord", "y_norm_coord"]
        self.gaze_partial_losses_val_mse = None
        self.gaze_partial_losses_val_l1 = None

    def _gaze_training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # determine individual losses
        individual_losses_mse = F.mse_loss(predictions["output_gaze"],
                                           batch["output_gaze"],
                                           reduction="none")
        individual_losses_mse = torch.mean(individual_losses_mse, dim=0)

        # log individual losses
        if not self.disable_write_to_disk:
            for n, l_mse in zip(self.gaze_names, individual_losses_mse):
                self.tb_writers[self.current_split].add_scalar(f"loss/train/output_gaze/{n}/mse", l_mse, global_step)

    def _gaze_validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # determine individual losses
        individual_losses_mse = F.mse_loss(predictions["output_gaze"],
                                           batch["output_gaze"],
                                           reduction="none")
        individual_losses_mse = torch.mean(individual_losses_mse, dim=0)
        individual_losses_l1 = F.l1_loss(predictions["output_gaze"],
                                         batch["output_gaze"],
                                         reduction="none")
        individual_losses_l1 = torch.mean(individual_losses_l1, dim=0)

        # accumulate individual losses
        if self.gaze_partial_losses_val_mse is None:
            self.gaze_partial_losses_val_mse = torch.zeros_like(individual_losses_mse)
            self.gaze_partial_losses_val_l1 = torch.zeros_like(individual_losses_l1)
        self.gaze_partial_losses_val_mse += individual_losses_mse
        self.gaze_partial_losses_val_l1 += individual_losses_l1

    def _gaze_validation_epoch_end(self, global_step, epoch, model, optimiser):
        # log individual losses
        if not self.disable_write_to_disk:
            for n, l_mse, l_l1 in zip(self.gaze_names, self.gaze_partial_losses_val_mse, self.gaze_partial_losses_val_l1):
                self.tb_writers[self.current_split].add_scalar(f"loss/val/output_gaze/{n}/mse",
                                                               l_mse / self.counter_val, global_step)
                self.tb_writers[self.current_split].add_scalar(f"loss/val/output_gaze/{n}/l1",
                                                               l_l1 / self.counter_val, global_step)

        # reset the loss accumulators
        self.gaze_partial_losses_val_mse = torch.zeros_like(self.gaze_partial_losses_val_mse)
        self.gaze_partial_losses_val_l1 = torch.zeros_like(self.gaze_partial_losses_val_l1)

    def training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        self._generic_training_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self._gaze_training_step_end(global_step, total_loss, partial_losses, batch, predictions)

    def validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        self._generic_validation_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self._gaze_validation_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self.counter_val += 1

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        self._generic_validation_epoch_end(global_step, epoch, model, optimiser)
        self._gaze_validation_epoch_end(global_step, epoch, model, optimiser)
        self.counter_val = 0

    def final_training_pass_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        raise NotImplementedError()

    def final_training_pass_epoch_end(self, global_step, epoch, model, optimiser):
        raise NotImplementedError()


class AttentionLogger(GenericLogger):

    def __init__(self, config, disable_write_to_disk=False):
        super().__init__(config, disable_write_to_disk)

        self.attention_partial_losses_val_kl = {}
        self.counter_val = 0

        self.log_attention_val = True

        self.loss_name = config["losses"]["output_attention"]

    def _attention_training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # if len(partial_losses) > 1:
        # log the partial losses
        if not self.disable_write_to_disk:
            for ln, l in partial_losses.items():
                if isinstance(l, dict) and "attention" in ln:
                    for pln, pl in l.items():
                        full_pln = "output_attention_{}".format(pln)
                        self.tb_writers[self.current_split].add_scalar(
                            f"loss/train/{full_pln}/{self.loss_name}", pl.item(), global_step)
                elif "attention" in ln:
                    self.tb_writers[self.current_split].add_scalar(
                        f"loss/train/{ln}/{self.loss_name}", l.item(), global_step)

    def _attention_validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # accumulate partial losses
        for ln, l in partial_losses.items():
            if isinstance(l, dict) and "attention" in ln:
                for pln, pl in l.items():
                    full_pln = "output_attention_{}".format(pln)
                    if full_pln not in self.attention_partial_losses_val_kl:
                        self.attention_partial_losses_val_kl[full_pln] = torch.zeros_like(pl)
                    self.attention_partial_losses_val_kl[full_pln] += pl
            elif "attention" in ln:
                if ln not in self.attention_partial_losses_val_kl:
                    self.attention_partial_losses_val_kl[ln] = torch.zeros_like(l)
                self.attention_partial_losses_val_kl[ln] += l

        if not self.disable_write_to_disk and self.log_attention_val:
            # get the original from the batch and the predictions and plot them
            # probably only include the original of the uncropped attention map...
            images_original = convert_attention_to_image(batch["original"]["output_attention"])

            attention_prediction = predictions["output_attention"]["final"] if isinstance(
                predictions["output_attention"], dict) else predictions["output_attention"]
            output_processing_func = {
                "kl": image_softmax,
                "ice": torch.sigmoid,
                "mse": lambda x: x,
            }[self.loss_name]
            images_prediction = convert_attention_to_image(output_processing_func(attention_prediction),
                                                           out_shape=images_original.shape[2:])

            self.tb_writers[self.current_split].add_images("attention/val/ground_truth", images_original,
                                                           global_step, dataformats="NCHW")
            self.tb_writers[self.current_split].add_images("attention/val/prediction", images_prediction,
                                                           global_step, dataformats="NCHW")

            self.log_attention_val = False

    def _attention_validation_epoch_end(self, global_step, epoch, model, optimiser):
        # log individual losses
        if not self.disable_write_to_disk:
            for ln, l in self.attention_partial_losses_val_kl.items():
                self.tb_writers[self.current_split].add_scalar(
                    f"loss/val/{ln}/{self.loss_name}", l.item() / self.counter_val, global_step)

        # reset the loss accumulators
        for ln, l in self.attention_partial_losses_val_kl.items():
            self.attention_partial_losses_val_kl[ln] = torch.zeros_like(l)

        self.log_attention_val = True

    def training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        self._generic_training_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self._attention_training_step_end(global_step, total_loss, partial_losses, batch, predictions)

    def validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        self._generic_validation_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self._attention_validation_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self.counter_val += 1

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        self._generic_validation_epoch_end(global_step, epoch, model, optimiser)
        self._attention_validation_epoch_end(global_step, epoch, model, optimiser)
        self.counter_val = 0

    def final_training_pass_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        raise NotImplementedError

    def final_training_pass_epoch_end(self, global_step, epoch, model, optimiser):
        raise NotImplementedError


class AttentionAndControlLogger(AttentionLogger, ControlLogger):

    def training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        self._generic_training_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self._attention_training_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self._control_training_step_end(global_step, total_loss, partial_losses, batch, predictions)

    def validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        self._generic_validation_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self._attention_validation_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self._control_validation_step_end(global_step, total_loss, partial_losses, batch, predictions)
        self.counter_val += 1

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        self._generic_validation_epoch_end(global_step, epoch, model, optimiser)
        self._attention_validation_epoch_end(global_step, epoch, model, optimiser)
        self._control_validation_epoch_end(global_step, epoch, model, optimiser)
        self.counter_val = 0

    def final_training_pass_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        raise NotImplementedError

    def final_training_pass_epoch_end(self, global_step, epoch, model, optimiser):
        raise NotImplementedError


class CVControlLogger(ControlLogger):

    def __init__(self, config, disable_write_to_disk=False):
        super().__init__(config, disable_write_to_disk)

        # TODO: "merge" this into the "normal" loggers

        # end-of-epoch training and validation error accumulators
        self.eoe_training_errors = None
        self.eoe_validation_errors = None

        # basically all the same stuff for training instead of validation
        self.total_loss_train_mse = None
        self.total_loss_train_l1 = None
        self.individual_losses_train_mse = None
        self.individual_losses_train_l1 = None
        self.counter_train = 0

    def update_info(self, **kwargs):
        super().update_info(**kwargs)

        if self.eoe_training_errors is None and "dataset" in kwargs:
            error_names = ["total"] + self.control_names
            self.eoe_training_errors = {
                f"split_{s}": {
                    n: {
                        "mse": [],
                        "l1": []
                    } for n in error_names
                } for s in range(self.splits)
            }
            self.eoe_validation_errors = {
                f"split_{s}": {
                    n: {
                        "mse": [],
                        "l1": []
                    } for n in error_names
                } for s in range(self.splits)
            }

        if "split" in kwargs:
            self.current_split = kwargs.get("split")

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        self.eoe_validation_errors[f"split_{self.current_split}"]["total"]["mse"].append(
            self.total_loss_val_mse.item() / self.counter_val)
        self.eoe_validation_errors[f"split_{self.current_split}"]["total"]["l1"].append(
            self.total_loss_val_l1.item() / self.counter_val)

        if not self.disable_write_to_disk:
            for n, l_mse, l_l1 in zip(self.control_names, self.control_partial_losses_val_mse,
                                      self.control_partial_losses_val_l1):
                self.eoe_validation_errors[f"split_{self.current_split}"][n]["mse"].append(l_mse.item() / self.counter_val)
                self.eoe_validation_errors[f"split_{self.current_split}"][n]["l1"].append(l_l1.item() / self.counter_val)

        super().validation_epoch_end(global_step, epoch, model, optimiser)
        self.counter_val += 1  # should actually be updated with update_info, but this doesn't really hurt

    def final_training_pass_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # determine individual losses
        individual_losses_mse = F.mse_loss(predictions["output_control"],
                                           batch["output_control"],
                                           reduction="none")
        individual_losses_mse = torch.mean(individual_losses_mse, dim=0)
        individual_losses_l1 = F.l1_loss(predictions["output_control"],
                                         batch["output_control"],
                                         reduction="none")
        individual_losses_l1 = torch.mean(individual_losses_l1, dim=0)

        # accumulate total loss
        if self.total_loss_train_mse is None:
            self.total_loss_train_mse = torch.zeros_like(total_loss)
            self.total_loss_train_l1 = torch.zeros_like(total_loss)
        self.total_loss_train_mse += total_loss
        self.total_loss_train_l1 += torch.mean(individual_losses_l1)

        # accumulate individual losses
        if self.individual_losses_train_mse is None:
            self.individual_losses_train_mse = torch.zeros_like(individual_losses_mse)
            self.individual_losses_train_l1 = torch.zeros_like(individual_losses_l1)
        self.individual_losses_train_mse += individual_losses_mse
        self.individual_losses_train_l1 += individual_losses_l1

        self.counter_train += 1

    def final_training_pass_epoch_end(self, global_step, epoch, model, optimiser):
        # record the errors
        self.eoe_training_errors[f"split_{self.current_split}"]["total"]["mse"].append(
            self.total_loss_train_mse.item() / self.counter_train)
        self.eoe_training_errors[f"split_{self.current_split}"]["total"]["l1"].append(
            self.total_loss_train_l1.item() / self.counter_train)

        if not self.disable_write_to_disk:
            for n, l_mse, l_l1 in zip(self.control_names, self.individual_losses_train_mse,
                                      self.individual_losses_train_l1):
                self.eoe_training_errors[f"split_{self.current_split}"][n]["mse"].append(l_mse.item() / self.counter_train)
                self.eoe_training_errors[f"split_{self.current_split}"][n]["l1"].append(l_l1.item() / self.counter_train)

        # reset the loss accumulators
        self.total_loss_train_mse = torch.zeros_like(self.total_loss_train_mse)
        self.total_loss_train_l1 = torch.zeros_like(self.total_loss_train_l1)
        self.individual_losses_train_mse = torch.zeros_like(self.individual_losses_train_mse)
        self.individual_losses_train_l1 = torch.zeros_like(self.individual_losses_train_l1)
        self.counter_train = 0

        # save stuff...
        if not self.disable_write_to_disk:
            if self.current_split == self.splits - 1:
                save_dict = {
                    "training_errors": self.eoe_training_errors,
                    "validation_errors": self.eoe_validation_errors
                }
                with open(os.path.join(self.log_dir, "cross_validation_results.json"), "w") as f:
                    json.dump(save_dict, f)
