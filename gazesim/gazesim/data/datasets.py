import os
import numpy as np
import pandas as pd
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from scipy.spatial.transform import Rotation
from gazesim.data.utils import get_indexed_reader, resolve_split_index_path, run_info_to_path, fps_reduction_index
from gazesim.data.transforms import MakeValidDistribution, ImageToAttentionMap, ManualRandomCrop
from gazesim.data.transforms import GaussianNoise, MultiRandomApply, DrEYEveTransform
from gazesim.data.constants import STATISTICS


class GenericDataset(Dataset):

    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        self.data_root = config["data_root"]
        # TODO: this needs to change if we want multiple possible outputs... should this be a list? or should
        #  there be specific inputs for control_gt, attention_gt etc. => I think this would be better actually
        self.split = split
        self.training = self.split == "train" and training
        self.fps = config["frames_per_second"]
        self.fps_reduced = self.fps != 60

        # load the index and data and select the subset to be used
        self.index = pd.read_csv(os.path.join(self.data_root, "index", "frame_index.csv"))

        # get information about the data split (train/test/val)
        if not isinstance(config["split_config"], str):
            # this should only happen when some "custom" splits are supposed to be used,
            # e.g. to "filter out" a single run to do predictions on that data
            split_index = config["split_config"]
        else:
            split_index = pd.read_csv(config["split_config"] + ".csv")
            if cv_split >= 0:
                split_index = split_index[f"split_{cv_split}"]
        self.index["split"] = split_index

        # TODO: subsampling here! (however, still need to account for the indexing change when getting frames)
        #  other possibility would be to just write frames multiple times... that would make things a lot easier
        #  however, this would e.g. make problems with optical flow...
        #  => should probably just adjust the "frame" column here => does this have other implications
        #  => yes, e.g. attention should actually still use the original frames I think; jesus what a mess
        #  => I guess that wouldn't be the case if the attention videos are generated again, but that might not be
        #     a good idea in terms of the time it would take
        #  => maybe should just calculate the new frames in one column and just use that => then what about indexing?
        #     could do something similar to the stacked frames (however, what would become of the whole frame stacking
        #     thing?)
        #  => easiest thing for existing stuff to work might be to reduce the dataframe, adjust the old frame column
        #     to index into the new videos (with lower FPS), and then just add a column with the original frames to
        #     be able to e.g. access attention => hopefully that way, stacking should also still be possible
        # if a video with FPS != 60 is supposed to be used, need to subsample the dataframe and "re-index"
        # the "frame" column (while maintaining the original to e.g. load from 60 FPS attention videos)
        subsampling_index = True
        if self.fps_reduced:
            subsampling_index, subsampling_index_numeric, new_frame_index = fps_reduction_index(
                self.index, fps=self.fps,
                groupby_columns=["track_name", "subject", "run"],
                return_sub_index_by_group=True
            )
            # keep the old frames for e.g. predicting attention, and compute new frames for low FPS indexing
            self.index["frame_original_fps"] = self.index["frame"].copy()
            self.index["frame"] = -1
            self.index.loc[subsampling_index_numeric, "frame"] = new_frame_index

        # TODO: need to think about whether subsampling can ever happen for attention as well (usually only masked
        #  videos should be subsampled/low FPS though)
        #  => this will be necessary if we want to use a lower frame rate for creating videos with the MPC

        # create the final sub-index and apply it to the dataframe
        self.sub_index = self.index["split"] == self.split
        self.sub_index = self.sub_index & subsampling_index
        self.index = self.index.loc[self.sub_index]
        self.index = self.index.reset_index(drop=True)

        # include the high-level labels as well
        self.index["label"] = 4
        for idx, (track_name, half) in enumerate([("flat", "left_half"), ("flat", "right_half"),
                                                  ("wave", "left_half"), ("wave", "right_half")]):
            self.index.loc[(self.index["track_name"] == track_name) & (self.index[half] == 1), "label"] = idx

    def __len__(self):
        return len(self.index.index)

    def __getitem__(self, item):
        return self.index.iloc[item]


class StackedGenericDataset(GenericDataset):

    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        super().__init__(config, split, cv_split, training)

        self.stack_size = config["stack_size"]

        # adjusting the index for stacking
        # 1. find contiguous sequences of frames
        sequences = []
        frames = self.index["frame"]
        jumps = (frames - frames.shift()) != 1
        frames = list(frames.index[jumps]) + [frames.index[-1] + 1]
        for i, start_index in enumerate(frames[:-1]):
            # note that the range [start_frame, end_frame) is exclusive
            sequences.append((start_index, frames[i + 1]))

        # 2. use second "index" that skips the (stack_size - 1) first frames of each contiguous sequence
        self.index["stack_index"] = -1
        df_col_index = self.index.columns.get_loc("stack_index")
        total_num_frame_stacks = 0
        for start_index, end_index in sequences:
            num_frames_seq = end_index - start_index
            num_frame_stacks = num_frames_seq - self.stack_size + 1
            if end_index - start_index < self.stack_size:
                continue

            # assign index over entire dataframe to "valid" frames
            index = np.arange(num_frame_stacks) + total_num_frame_stacks
            self.index.iloc[(start_index + self.stack_size - 1):end_index, df_col_index] = index

            # keep track of the current number of stacks (highest index)
            total_num_frame_stacks += num_frame_stacks

    def __len__(self):
        return self.index["stack_index"].max() + 1

    def __getitem__(self, item):
        current_stack_index = self.index.index[self.index["stack_index"] == item].values[0]
        current_stack_df = self.index.iloc[(current_stack_index - self.stack_size + 1):(current_stack_index + 1)]
        return current_stack_df.values


class ToControlDataset(StackedGenericDataset):

    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        super().__init__(config, split, cv_split, training)

        self.control_output_name = config["control_ground_truth"]
        self.output_columns = []

        # load additional information and put it in the index
        ground_truth = pd.read_csv(os.path.join(self.data_root, "index", f"{self.control_output_name}.csv"))
        ground_truth = ground_truth.loc[self.sub_index]
        ground_truth = ground_truth.reset_index(drop=True)
        for col in ground_truth:
            self.index[col] = ground_truth[col]
            self.output_columns.append(col)

        self.output_normalisation = None
        if config["control_normalisation"]:
            cnr_dict = {}
            for col in self.output_columns:
                if col in config["control_normalisation_range"]:
                    cnr_dict[col] = config["control_normalisation_range"][col]
                else:
                    cnr_dict[col] = 1.0
            self.output_normalisation = np.array([cnr_dict[col] for col in self.output_columns])

    def _get_control(self, item):
        # TODO: this might not actually work for the other datasets anymore if this class subclasses the stacked DS
        #  => would it be enough to set the default stack size to 0?
        current_stack_index = self.index.index[self.index["stack_index"] == item].values[0]
        # current_stack_index = item

        # extract the control GT from the dataframe
        control = self.index[self.output_columns].iloc[current_stack_index].values
        if self.output_normalisation is not None:
            control /= self.output_normalisation
        control = torch.from_numpy(control).float()

        return control


class ToAttentionDataset(GenericDataset):

    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        super().__init__(config, split, cv_split, training)

        self.attention_output_name = config["attention_ground_truth"]

        self.random_crop = None
        size_before_crop = config["resize"]
        if self.training and config["video_random_cropping"] and config["resize"] != -1:
            if not isinstance(config["resize"], tuple):
                height = config["resize"]
                width = int(800 / (600 / height))
                config["resize"] = (height, width)
            size_before_crop = tuple((np.array(config["resize"]) * config["vrc_factor_before_crop"]).astype(int))
            self.random_crop = ManualRandomCrop(size_before_crop, config["resize"])

        self.attention_output_transform = transforms.Compose([
            ImageToAttentionMap(),
            transforms.ToPILImage(),
            transforms.Resize(size_before_crop),
            self.random_crop if self.random_crop is not None else lambda x: x,
            transforms.ToTensor(),
            MakeValidDistribution(),
        ])

        self.video_output_readers = {}

    def _get_attention(self, item):
        current_row = self.index.iloc[item]
        current_frame_index = current_row["frame_original_fps" if self.fps_reduced else "frame"]
        current_run_dir = os.path.join(self.data_root, run_info_to_path(current_row["subject"],
                                                                        current_row["run"],
                                                                        current_row["track_name"]))

        # initialise the video readers if necessary
        if current_run_dir not in self.video_output_readers:
            self.video_output_readers[current_run_dir] = {
                "output_attention": get_indexed_reader(os.path.join(current_run_dir, f"{self.attention_output_name}.mp4"))
            }

        # read the frame
        attention = self.video_output_readers[current_run_dir]["output_attention"][current_frame_index]

        # original and transformed
        attention_original = np.array(attention.copy())
        attention = self.attention_output_transform(attention)

        return attention, attention_original

    def __getitem__(self, item):
        attention, attention_original = self._get_attention(item)
        label = self.index["label"].iloc[item]

        out = {
            # original
            "original": {"output_attention": attention_original},
            # transformed
            "output_attention": attention,
            "label_high_level": label,
        }

        return out


class ToGazeDataset(StackedGenericDataset):

    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        super().__init__(config, split, cv_split, training)

        self.output_columns = []

        # load additional information and put it in the index
        ground_truth = pd.read_csv(os.path.join(self.data_root, "index", "{}.csv".format(
            config.get("gaze_ground_truth", "frame_mean_gaze_gt"))))
        ground_truth = ground_truth.loc[self.sub_index]
        ground_truth = ground_truth.reset_index(drop=True)
        for col in ground_truth:
            self.index[col] = ground_truth[col]
            self.output_columns.append(col)

        self.output_clipping = config["clip_gaze"]
        if self.output_clipping:
            for col in self.output_columns:
                self.index[col] = self.index[col].clip(-1.0, 1.0)

        self.output_scaling = config["scale_gaze"]
        if self.output_scaling:
            # TODO: might want to change this since now the range is twice as large as intended...
            #  => need to be consistent with how everything is handled in DDA though
            self.index[self.output_columns] *= np.array([800.0, 600.0])

    def _get_gaze(self, item):
        current_stack_index = self.index.index[self.index["stack_index"] == item].values[0]

        # extract the control GT from the dataframe
        gaze = self.index[self.output_columns].iloc[current_stack_index].values
        gaze = torch.from_numpy(gaze).float()

        return gaze

    def __getitem__(self, item):
        gaze = self._get_gaze(item)
        label = self.index["label"].iloc[item]
        out = {"output_gaze": gaze, "label_high_level": label}
        return out


class ImageDataset(GenericDataset):

    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        super().__init__(config, split, cv_split, training)

        self.video_input_names = config["input_video_names"]

        # TODO: if there are multiple videos, should probably be able to specify (or decide on the fly?) whether
        #  they should be loaded at 60 FPS or less? => for now, just assume that only videos with the same
        #  frame rate (except for the attention output) are used

        """
        with open(config["split_config"] + "_info.json", "r") as f:
            split_index_info = json.load(f)
        input_statistics = {}
        for i in self.video_input_names:
            if config["no_normalisation"]:
                input_statistics[i] = {"mean": np.array([0, 0, 0]), "std": np.array([1, 1, 1])}
            else:
                if "mean" in split_index_info and "std" in split_index_info:
                    input_statistics[i] = {
                        "mean": split_index_info["mean"][i] if i in split_index_info["mean"] else STATISTICS["mean"][i],
                        "std": split_index_info["std"][i] if i in split_index_info["std"] else STATISTICS["std"][i]
                    }
                else:
                    input_statistics[i] = {
                        "mean": STATISTICS["mean"][i],
                        "std": STATISTICS["std"][i]
                    }
        """

        size_before_crop = config["resize"]
        if not hasattr(self, "random_crop"):
            self.random_crop = None
            if self.training and config["video_random_cropping"] and config["resize"] != -1:
                if not isinstance(config["resize"], tuple):
                    height = config["resize"]
                    width = int(800 / (600 / height))
                    config["resize"] = (height, width)
                size_before_crop = tuple((np.array(config["resize"]) * config["vrc_factor_before_crop"]).astype(int))
                self.random_crop = ManualRandomCrop(size_before_crop, config["resize"])
        elif self.random_crop is not None:
            size_before_crop = tuple((np.array(config["resize"]) * 1.5).astype(int))

        input_statistics = {}
        for i in self.video_input_names:
            input_statistics[i] = {"mean": np.array([0.0, 0.0, 0.0]), "std": np.array([1.0, 1.0, 1.0])}
        self.video_input_transforms = [transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size_before_crop),
            self.random_crop if self.random_crop is not None else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
        ]) for i in self.video_input_names]

        self.video_input_augmentation = None
        if self.training and config["video_data_augmentation"]:
            jitter = config["vda_jitter_range"]
            self.video_input_augmentation = [MultiRandomApply([
                transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter),
                GaussianNoise(config["vda_gaussian_noise_sigma"]),
                # TODO: S&P noise (?): https://stackoverflow.com/a/30609854
                transforms.GaussianBlur(11),
                transforms.RandomErasing(1.0)
            ], p=config["vda_probability"]) for _ in self.video_input_names]

        self.video_input_readers = {}

        # to be used by subclasses mostly
        self.return_original = config.get("return_original", False)

    def _get_image(self, item):
        current_row = self.index.iloc[item]
        current_frame_index = current_row["frame"]
        current_run_dir = os.path.join(self.data_root, run_info_to_path(current_row["subject"],
                                                                        current_row["run"],
                                                                        current_row["track_name"]))

        # initialise the video readers if necessary
        if current_run_dir not in self.video_input_readers:
            self.video_input_readers[current_run_dir] = {
                f"input_image_{idx}": get_indexed_reader(os.path.join(current_run_dir, f"{i}.mp4"))
                for idx, i in enumerate(self.video_input_names)
            }

        # read the frames
        image = [self.video_input_readers[current_run_dir][f"input_image_{idx}"][current_frame_index]
                 for idx in range(len(self.video_input_names))]

        # original and non-original
        image_original = [np.array(i.copy()) for i in image]
        image = [self.video_input_transforms[idx](i) for idx, i in enumerate(image)]
        if self.training and self.video_input_augmentation is not None:
            image = [self.video_input_augmentation[idx](i) for idx, i in enumerate(image)]

        return image, image_original

    def __getitem__(self, item):
        image, image_original = self._get_image(item)
        label = self.index["label"].iloc[item]

        # original
        out = {"original": {f"input_image_{idx}": i for idx, i in enumerate(image_original)}}

        # transformed
        for idx, i in enumerate(image):
            out[f"input_image_{idx}"] = i
        out["label_high_level"] = label

        return out


class StackedImageDataset(ImageDataset):

    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        super().__init__(config, split, cv_split, training)

        self.stack_size = config["stack_size"]
        self.dreyeve_transforms = config["dreyeve_transforms"]

        # adjusting the index for stacking
        # 1. find contiguous sequences of frames
        sequences = []
        frames = self.index["frame"]
        jumps = (frames - frames.shift()) != 1
        frames = list(frames.index[jumps]) + [frames.index[-1] + 1]
        for i, start_index in enumerate(frames[:-1]):
            # note that the range [start_frame, end_frame) is exclusive
            sequences.append((start_index, frames[i + 1]))

        # 2. use second "index" that skips the (stack_size - 1) first frames of each contiguous sequence
        self.index["stack_index"] = -1
        df_col_index = self.index.columns.get_loc("stack_index")
        total_num_frame_stacks = 0
        for start_index, end_index in sequences:
            num_frames_seq = end_index - start_index
            num_frame_stacks = num_frames_seq - self.stack_size + 1
            if end_index - start_index < self.stack_size:
                continue

            # assign index over entire dataframe to "valid" frames
            index = np.arange(num_frame_stacks) + total_num_frame_stacks
            self.index.iloc[(start_index + self.stack_size - 1):end_index, df_col_index] = index

            # keep track of the current number of stacks (highest index)
            total_num_frame_stacks += num_frame_stacks

    def __len__(self):
        return self.index["stack_index"].max() + 1

    def _get_image_stack(self, item):
        current_stack_index = self.index.index[self.index["stack_index"] == item].values[0]
        current_stack_df = self.index.iloc[(current_stack_index - self.stack_size + 1):(current_stack_index + 1)]
        current_frame_index = current_stack_df["frame"].iloc[-1]
        current_run_dir = os.path.join(self.data_root, run_info_to_path(current_stack_df["subject"].iloc[0],
                                                                        current_stack_df["run"].iloc[0],
                                                                        current_stack_df["track_name"].iloc[0]))

        # initialise the video readers if necessary
        if current_run_dir not in self.video_input_readers:
            self.video_input_readers[current_run_dir] = {
                f"input_image_{idx}": get_indexed_reader(os.path.join(current_run_dir, f"{i}.mp4"))
                for idx, i in enumerate(self.video_input_names)
            }

        image_stack = [[] for _ in range(len(self.video_input_names))]
        for idx in range(len(self.video_input_names)):
            for in_stack_idx, (_, row) in enumerate(current_stack_df.iterrows()):
                image = self.video_input_readers[current_run_dir][f"input_image_{idx}"][row["frame"]]
                if not self.dreyeve_transforms:
                    image = self.video_input_transforms[idx](image)
                    if self.video_input_augmentation is not None:  # maybe don't apply here for dreyeve?
                        # TODO: think about whether these should only be applied after the images have been stacked
                        #  (according to the PyTorch documentation that should mean that the same transform is applied
                        #  to all images in the "batch", which might be desirable if this doesn't seem to work, might
                        #  e.g. make more sense conceptually with random masking)
                        image = self.video_input_augmentation[idx](image)
                image_stack[idx].append(image)

            if not self.dreyeve_transforms:
                image_stack[idx] = np.stack(image_stack[idx], 1)

        # TODO: think about whether we want original, but probably not, just because of the size
        return image_stack


class StateDataset(GenericDataset):

    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        super().__init__(config, split, cv_split, training)

        self.state_input_names = config["drone_state_names"]

        # load additional information and put it in the index
        drone_state = pd.read_csv(os.path.join(self.data_root, "index", "drone_state_original.csv"))
        drone_state = drone_state.loc[self.sub_index]
        drone_state = drone_state.reset_index(drop=True)
        for col in self.state_input_names:
            self.index[col] = drone_state[col]

    def _get_state(self, item):
        # read the state
        state = self.index[self.state_input_names].iloc[item].values
        state = torch.from_numpy(state).float()

        return state


# TODO: should all of the stuff below just be stacked by default and also, should I just change everything
#  to be stacked and if there's only one frame, you can just squeeze out that extra dimension?
class FeatureTrackDataset(StackedGenericDataset):

    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        super().__init__(config, split, cv_split, training)

        self.feature_track_name = config["feature_track_name"]
        self.feature_track_num = config["feature_track_num"]
        # self.feature_track_sampling_mode = config["feature_track_sampling_mode"]

        self.feature_track_readers = {}

    def _get_feature_tracks(self, item):
        current_stack_index = self.index.index[self.index["stack_index"] == item].values[0]
        current_stack_df = self.index.iloc[(current_stack_index - self.stack_size + 1):(current_stack_index + 1)]
        current_run_dir = os.path.join(self.data_root, run_info_to_path(current_stack_df["subject"].iloc[0],
                                                                        current_stack_df["run"].iloc[0],
                                                                        current_stack_df["track_name"].iloc[0]))

        # current_row = self.index.iloc[item]
        # current_frame_index = current_row["frame"]
        # current_run_dir = os.path.join(self.data_root, run_info_to_path(current_row["subject"],
        #                                                                 current_row["run"],
        #                                                                 current_row["track_name"]))

        # initialise the numpy file handles if necessary
        if current_run_dir not in self.feature_track_readers:
            self.feature_track_readers[current_run_dir] = np.load(os.path.join(
                current_run_dir, "{}.npz".format(self.feature_track_name)))

        # read the feature track information
        feature_track_stack = []
        for in_stack_idx, (_, row) in enumerate(current_stack_df.iterrows()):
            feature_tracks = self.feature_track_readers[current_run_dir]["arr_{}".format(row["frame"])]
            # TODO: if FPS subsampling is supposed to work here, need to use the original frame I think...
            #  but that would be conceptually erroneous

            # sample from the feature tracks
            # 3 cases:
            # - fewer features than specified to use
            #   => can either sample with replacement or take all and then sample the rest
            # - more features than specified to use
            #   => can either sample all or take first X
            if feature_tracks.shape[0] == 0:
                # no features at all
                # => everything needs to be 0 or something like that probably
                feature_tracks = np.zeros((self.feature_track_num, 6))
            else:
                num_missing = self.feature_track_num - feature_tracks.shape[0]
                if num_missing > 0:
                    idx = np.random.choice(range(feature_tracks.shape[0]), size=num_missing)
                    feature_tracks = np.concatenate([feature_tracks, feature_tracks[idx]])
                elif num_missing < 0:
                    idx = np.random.choice(range(feature_tracks.shape[0]), size=self.feature_track_num, replace=False)
                    feature_tracks = feature_tracks[idx]

            # leave out the IDs
            feature_tracks = feature_tracks[:, 1:]

            feature_track_stack.append(feature_tracks)

        # stack the feature tracks or reduce in dimensionality
        if self.stack_size == 1:
            feature_tracks = torch.from_numpy(feature_track_stack[0]).float()
        else:
            shapes = [tuple(fts.shape) for fts in feature_track_stack]
            if len(set(shapes)) != 1:
                print(shapes)
            feature_tracks = np.stack(feature_track_stack, axis=0)
            feature_tracks = torch.from_numpy(feature_tracks).float()
            # TODO: might have to reorder dimensions for DDA input

        # TODO: maybe normalisation (of "tracked frame count") as done in DDA?

        return feature_tracks

    def __getitem__(self, item):
        feature_tracks = self._get_feature_tracks(item)
        if self.stack_size == 1:
            out = {"input_feature_tracks": feature_tracks}
        else:
            out = {"input_feature_tracks": {"stack": feature_tracks}}
        return out


class ReferenceDataset(StackedGenericDataset):
    # how is this different from StateDataset? not really sure... might even replace that
    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        super().__init__(config, split, cv_split, training)

        self.reference_name = config["reference_name"]
        self.reference_variables = config["reference_variables"]

        # determine whether to convert to rotation matrix
        self.convert_rotation = False
        quaternion_variables = [f"rotation_{r}" for r in "xyzw"]
        if all([qv in self.reference_variables for qv in quaternion_variables]):
            self.reference_variables = [rv for rv in self.reference_variables
                                        if rv not in quaternion_variables] + quaternion_variables
            self.convert_rotation = True

        # load additional information and put it in the index
        drone_reference = pd.read_csv(os.path.join(self.data_root, "index", "{}.csv".format(self.reference_name)))
        drone_reference = drone_reference.loc[self.sub_index]
        drone_reference = drone_reference.reset_index(drop=True)
        for col in self.reference_variables:
            self.index[col] = drone_reference[col]

    def _get_reference(self, item):
        current_stack_index = self.index.index[self.index["stack_index"] == item].values[0]
        current_stack_df = self.index.iloc[(current_stack_index - self.stack_size + 1):(current_stack_index + 1)]

        # read the reference
        reference = current_stack_df[self.reference_variables].values

        if self.convert_rotation:
            # convert to (flattened) rotation matrix
            reference_rot = Rotation.from_quat(reference[:, -4:]).as_matrix().reshape((-1, 9))

            # update reference
            reference = np.concatenate((reference[:, :-4], reference_rot), axis=1)

        # convert to torch tensor
        reference = torch.from_numpy(reference).float()

        return reference


class StateEstimateDataset(StackedGenericDataset):

    NOISE_STD = {
        "position_x": 0.5,
        "position_y": 0.5,
        "position_z": 0.1,
        "velocity_x": 0.3,
        "velocity_y": 0.3,
        "velocity_z": 0.1,
        "rotation_w": 0.1,
        "rotation_x": 0.05,
        "rotation_y": 0.05,
        "rotation_z": 0.1,
        "omega_x": 0.1,
        "omega_y": 0.1,
        "omega_z": 0.1,
    }

    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        super().__init__(config, split, cv_split, training)

        self.state_estimate_name = config["state_estimate_name"]
        self.state_estimate_variables = config["state_estimate_variables"]
        self.state_estimate_data_augmentation = config["state_estimate_data_augmentation"]

        # determine whether to convert to rotation matrix
        self.convert_rotation = False
        quaternion_variables = [f"rotation_{r}" for r in "xyzw"]
        if all([qv in self.state_estimate_variables for qv in quaternion_variables]):
            self.state_estimate_variables = [sev for sev in self.state_estimate_variables
                                             if sev not in quaternion_variables] + quaternion_variables
            self.convert_rotation = True

        self.noise_std = None
        if self.state_estimate_data_augmentation:
            self.noise_std = [StateEstimateDataset.NOISE_STD[sev] for sev in self.state_estimate_variables]

        # load additional information and put it in the index
        drone_state_estimate = pd.read_csv(os.path.join(self.data_root,
                                                        "index", "{}.csv".format(self.state_estimate_name)))
        drone_state_estimate = drone_state_estimate.loc[self.sub_index]
        drone_state_estimate = drone_state_estimate.reset_index(drop=True)
        for col in self.state_estimate_variables:
            self.index[col] = drone_state_estimate[col]

    def _get_state_estimate(self, item):
        current_stack_index = self.index.index[self.index["stack_index"] == item].values[0]
        current_stack_df = self.index.iloc[(current_stack_index - self.stack_size + 1):(current_stack_index + 1)]

        # read the state estimate
        state_estimate = current_stack_df[self.state_estimate_variables].values

        # this should obviously be dependent on which state is used as input...
        if self.state_estimate_data_augmentation:
            state_estimate += np.random.normal(loc=0.0, scale=self.noise_std, size=state_estimate.shape)

        if self.convert_rotation:
            # convert to (flattened) rotation matrix
            state_estimate_rot = Rotation.from_quat(state_estimate[:, -4:]).as_matrix().reshape((-1, 9))

            # update state estimate
            state_estimate = np.concatenate((state_estimate[:, :-4], state_estimate_rot), axis=1)

        # convert to torch tensor
        state_estimate = torch.from_numpy(state_estimate).float()

        return state_estimate


class DDADataset(FeatureTrackDataset, ReferenceDataset, StateEstimateDataset, ToControlDataset):

    def __getitem__(self, item):
        feature_tracks = self._get_feature_tracks(item)
        reference = self._get_reference(item)
        state_estimate = self._get_state_estimate(item)
        control = self._get_control(item)

        # concatenate reference and state estimate into one input
        state = torch.cat((reference, state_estimate), dim=-1)

        # reshape the tensors to match what is for the DDA network (i.e. "channel" dimension in right place)
        feature_tracks = feature_tracks.permute(2, 0, 1)
        state = state.permute(1, 0)

        out = {
            "input_feature_tracks": {"stack": feature_tracks},
            "input_state": {"stack": state},
            "output_control": control
        }
        return out


class StateToControlDataset(StateDataset, ToControlDataset):

    def __getitem__(self, item):
        state = self._get_state(item)
        control = self._get_control(item)
        label = self.index["label"].iloc[item]

        # no need to store any original stuff
        out = {
            "input_state": state,
            "output_control": control,
            "label_high_level": label
        }

        return out


class ImageToAttentionDataset(ImageDataset, ToAttentionDataset):

    def __getitem__(self, item):
        image, image_original = self._get_image(item)
        attention, attention_original = self._get_attention(item)
        label = self.index["label"].iloc[item]

        if self.random_crop is not None:
            self.random_crop.update()

        # original
        out = {"original": {}}
        if self.return_original:
            out["original"] = {f"input_image_{idx}": i for idx, i in enumerate(image_original)}
        out["original"]["output_attention"] = attention_original

        # transformed
        for idx, i in enumerate(image):
            out[f"input_image_{idx}"] = i
        out["output_attention"] = attention
        out["label_high_level"] = label

        return out


class ImageToControlDataset(ImageDataset, ToControlDataset):

    def __getitem__(self, item):
        image, image_original = self._get_image(item)
        control = self._get_control(item)
        label = self.index["label"].iloc[item]

        # original
        out = {}
        if self.return_original:
            out["original"] = {f"input_image_{idx}": i for idx, i in enumerate(image_original)}

        # transformed
        for idx, i in enumerate(image):
            out[f"input_image_{idx}"] = i
        out["output_control"] = control
        out["label_high_level"] = label

        return out


class ImageToGazeDataset(ImageDataset, ToGazeDataset):

    def __getitem__(self, item):
        if self.random_crop is not None:
            self.random_crop.update()

        image, image_original = self._get_image(item)
        gaze = self._get_gaze(item)
        label = self.index["label"].iloc[item]

        # original
        out = {}
        if self.return_original:
            out["original"] = {f"input_image_{idx}": i for idx, i in enumerate(image_original)}

        # transformed
        for idx, i in enumerate(image):
            out[f"input_image_{idx}"] = i
        out["output_gaze"] = gaze
        out["label_high_level"] = label

        return out


class ImageAndStateToControlDataset(ImageDataset, StateDataset, ToControlDataset):

    def __getitem__(self, item):
        image, image_original = self._get_image(item)
        state = self._get_state(item)
        control = self._get_control(item)
        label = self.index["label"].iloc[item]

        # original
        out = {}
        if self.return_original:
            out["original"] = {f"input_image_{idx}": i for idx, i in enumerate(image_original)}

        # transformed
        for idx, i in enumerate(image):
            out[f"input_image_{idx}"] = i
        out["input_state"] = state
        out["output_control"] = control
        out["label_high_level"] = label

        return out


class ImageToAttentionAndControlDataset(ImageDataset, ToAttentionDataset, ToControlDataset):

    def __getitem__(self, item):
        image, image_original = self._get_image(item)
        attention, attention_original = self._get_attention(item)
        control = self._get_control(item)
        label = self.index["label"].iloc[item]

        # original
        out = {"original": {}}
        if self.return_original:
            out["original"] = {f"input_image_{idx}": i for idx, i in enumerate(image_original)}
        out["original"]["output_attention"] = attention_original

        # transformed
        for idx, i in enumerate(image):
            out[f"input_image_{idx}"] = i
        out["output_attention"] = attention
        out["output_control"] = control
        out["label_high_level"] = label

        return out


class ImageAndStateToAttentionAndControlDataset(ImageDataset, StateDataset, ToAttentionDataset, ToControlDataset):

    def __getitem__(self, item):
        image, image_original = self._get_image(item)
        state = self._get_state(item)
        attention, attention_original = self._get_attention(item)
        control = self._get_control(item)
        label = self.index["label"].iloc[item]

        # original
        out = {"original": {f"input_image_{idx}": i for idx, i in enumerate(image_original)}}
        out["original"]["output_attention"] = attention_original

        # transformed
        for idx, i in enumerate(image):
            out[f"input_image_{idx}"] = i
        out["input_state"] = state
        out["output_attention"] = attention
        out["output_control"] = control
        out["label_high_level"] = label

        return out


class StackedImageToControlDataset(StackedImageDataset, ToControlDataset):

    def __getitem__(self, item):
        image_stack = self._get_image_stack(item)
        control = self._get_control(item)
        label = self.index["label"].iloc[item]

        # original
        # out = {"original": {f"input_image_{idx}": i for idx, i in enumerate(image_original)}}
        out = {}

        # transformed
        for idx, i in enumerate(image_stack):
            out[f"input_image_{idx}"] = {"stack": i}
        out["output_control"] = control
        out["label_high_level"] = label

        return out


class StackedImageAndStateToControlDataset(StackedImageDataset, StateDataset, ToControlDataset):

    def __getitem__(self, item):
        image_stack = self._get_image_stack(item)
        state = self._get_state(item)
        control = self._get_control(item)
        label = self.index["label"].iloc[item]

        # original
        # out = {"original": {f"input_image_{idx}": i for idx, i in enumerate(image_original)}}
        out = {}

        # transformed
        for idx, i in enumerate(image_stack):
            out[f"input_image_{idx}"] = {"stack": i}
        out["input_state"] = state
        out["output_control"] = control
        out["label_high_level"] = label

        return out


class StackedImageToAttentionDataset(Dataset):

    def __init__(self, config, split, cv_split=-1):
        super().__init__()
        self.data_root = config["data_root"]
        self.input_names = config["input_video_names"]
        self.output_name = config["ground_truth_name"]
        self.split = split
        self.stack_size = config["stack_size"]
        self.dreyeve_transforms = config["dreyeve_transforms"]

        frame_index = pd.read_csv(os.path.join(self.data_root, "index", "frame_index.csv"))
        split_index = pd.read_csv(config["split_config"] + ".csv")
        if cv_split >= 0:
            split_index = split_index[f"split_{cv_split}"]
        frame_index["split"] = split_index
        self.index = frame_index[frame_index["split"] == self.split].copy().reset_index(drop=True)

        with open(config["split_config"] + "_info.json", "r") as f:
            split_index_info = json.load(f)
        input_statistics = {}
        for i in self.input_names:
            if config["no_normalisation"]:
                input_statistics[i] = {"mean": np.array([0, 0, 0]), "std": np.array([1, 1, 1])}
            else:
                if "mean" in split_index_info and "std" in split_index_info:
                    input_statistics[i] = {
                        "mean": split_index_info["mean"][i] if i in split_index_info["mean"] else STATISTICS["mean"][i],
                        "std": split_index_info["std"][i] if i in split_index_info["std"] else STATISTICS["std"][i]
                    }
                else:
                    input_statistics[i] = {
                        "mean": STATISTICS["mean"][i],
                        "std": STATISTICS["std"][i]
                    }

        self.random_crop = None
        if self.dreyeve_transforms:
            # define the following transforms
            # - loaded stack to 112x112
            # - load last frame to 448x448
            # - loaded stack to 256x256 => random crop to 112x112
            self.random_crop = ManualRandomCrop((256, 256), (112, 112))

            self.input_transforms = {
                "stack": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.input_names],
                "stack_crop": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    self.random_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"]),
                ]) for i in self.input_names],
                "last_frame": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.input_names],
            }

            self.output_transforms = {
                "stack": transforms.Compose([
                    ImageToAttentionMap(),
                    transforms.ToPILImage(),
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    MakeValidDistribution()
                ]),
                "stack_crop": transforms.Compose([
                    ImageToAttentionMap(),
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    self.random_crop,
                    transforms.ToTensor(),
                    MakeValidDistribution(),
                    # TODO: really not sure if this is proper to do (in general and particularly in this case
                    #  - in general: might actually want to learn to predict in such a way that (usually) the
                    #    upscaled output is close-ish to a valid distribution, which it might not even be if
                    #    the small version is already one
                    #  - here specifically: this might actually make problems because there will be cases where
                    #    every value is 0 and maybe cases where very few values will be non-zero => in the latter
                    #    case these values would be adjusted to be very high and I'm unsure if that's a good thing
                    #  - on the other hand: not sure if KL-divergence works (not just conceptually, but also
                    #    in practice) if this isn't properly normalised => if it doesn't work in practice, might
                    #    still want to consider not doing this normalisation when using MSE (if that ever happens)
                ])
            }
        else:
            # just a simple transform for every frame
            self.input_transforms = {
                "stack": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(config["resize"]),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.input_names]
            }
            self.output_transforms = {
                "stack": transforms.Compose([
                    ImageToAttentionMap(),
                    transforms.ToPILImage(),
                    transforms.Resize(config["resize"]),
                    transforms.ToTensor(),
                    MakeValidDistribution(),
                ])
            }
        # TODO: MIGHT NEED TO ADD ADDITIONAL STREAM/TRANSFORM/WHATEVER FOR
        #  THE CROPPED STREAM OF THE COARSE-REFINE ARCHITECTURE

        self.video_readers = {}
        """
        self.index["run_dir"] = None
        unique_run_info = self.index.groupby(["track_name", "subject", "run"]).size().reset_index()
        for _, row in unique_run_info.iterrows():
            run_dir = os.path.join(self.data_root, run_info_to_path(row["subject"], row["run"], row["track_name"]))
            if run_dir not in self.video_readers:
                self.video_readers[run_dir] = {
                    f"input_stack_{idx}": get_indexed_reader(os.path.join(run_dir, f"{i}.mp4"))
                    for idx, i in enumerate(self.input_names)
                }
                self.video_readers[run_dir]["output_attention"] = get_indexed_reader(os.path.join(
                    run_dir, f"{self.output_name}.mp4"))
            self.index.loc[(self.index["track_name"] == row["track_name"])
                           & (self.index["subject"] == row["subject"])
                           & (self.index["run"] == row["run"]), "run_dir"] = run_dir
        """

        # find contiguous sequences of frames
        sequences = []
        frames = self.index["frame"]
        jumps = (frames - frames.shift()) != 1
        frames = list(frames.index[jumps]) + [frames.index[-1] + 1]
        for i, start_index in enumerate(frames[:-1]):
            # note that the range [start_frame, end_frame) is exclusive
            sequences.append((start_index, frames[i + 1]))

        # need some way to not mess up the indexing, but also cannot remove data
        # maybe use second "index" that skips the (stack_size - 1) first frames of each contiguous sequence?
        self.index["stack_index"] = -1
        df_col_index = self.index.columns.get_loc("stack_index")
        total_num_frame_stacks = 0
        for start_index, end_index in sequences:
            num_frames_seq = end_index - start_index
            num_frame_stacks = num_frames_seq - self.stack_size + 1
            if end_index - start_index < self.stack_size:
                continue

            # assign index over entire dataframe to "valid" frames
            index = np.arange(num_frame_stacks) + total_num_frame_stacks
            # print(len(index))
            # print(end_index - (start_index + self.stack_size - 1))
            # print(len(self.index.iloc[(start_index + self.stack_size - 1):end_index, df_col_index].index))
            self.index.iloc[(start_index + self.stack_size - 1):end_index, df_col_index] = index

            # keep track of the current number of stacks (highest index)
            total_num_frame_stacks += num_frame_stacks

    def __len__(self):
        return self.index["stack_index"].max() + 1

    def __getitem__(self, item):
        # get the information about the current item
        # TODO: maybe just change the index of the dataframe? might make this particular step easier
        current_stack_index = self.index.index[self.index["stack_index"] == item].values[0]
        # current_run_dir = self.index["run_dir"].iloc[current_stack_index]
        current_stack_df = self.index.iloc[(current_stack_index - self.stack_size + 1):(current_stack_index + 1)]
        current_frame_index = current_stack_df["frame"].iloc[-1]
        current_run_dir = os.path.join(self.data_root, run_info_to_path(current_stack_df["subject"].iloc[0],
                                                                        current_stack_df["run"].iloc[0],
                                                                        current_stack_df["track_name"].iloc[0]))

        # initialise the video readers if necessary
        if current_run_dir not in self.video_readers:
            # print(os.path.join(current_run_dir, f"{self.video_input_names[1]}.mp4"))
            self.video_readers[current_run_dir] = {
                f"input_image_{idx}": get_indexed_reader(os.path.join(current_run_dir, f"{i}.mp4"))
                for idx, i in enumerate(self.input_names)
            }
            self.video_readers[current_run_dir]["output_attention"] = get_indexed_reader(os.path.join(
                current_run_dir, f"{self.output_name}.mp4"))

        # prepare new "crop parameters" to get a random crop consistent across different tensors
        if self.random_crop is not None:
            self.random_crop.update()

        # read the frames
        inputs = [{k: [] for k in self.input_transforms} for _ in self.input_names]
        # inputs_original = [{} for _ in self.input_names]
        for idx, _ in enumerate(self.input_names):
            for in_stack_idx, (_, row) in enumerate(current_stack_df.iterrows()):
                frame = self.video_readers[current_run_dir][f"input_image_{idx}"][row["frame"]]
                # frame_original = frame.copy()

                # apply input transform here already
                for k in self.input_transforms:
                    # TODO: for stack_crop, need to store random crop parameters and also apply them to the output
                    #  for that image... => any point in rethinking how transforms are applied? e.g. having custom
                    #  class (e.g. DrEYEveTransform) that applies everything, including the correct random crop
                    #  (see here: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
                    if k != "last_frame":
                        inputs[idx][k].append(self.input_transforms[k][idx](frame))
                    elif in_stack_idx == self.stack_size - 1:
                        inputs[idx][k] = self.input_transforms[k][idx](frame)
                # frame = self.input_transforms[idx](frame)
                # inputs[idx].append(frame)
                # inputs_original[idx].append(frame_original)

            # stack the frames
            for k in self.input_transforms:
                if k != "last_frame":
                    inputs[idx][k] = torch.stack(inputs[idx][k], 1)
            # inputs[idx] = torch.stack(inputs[idx], 1)
            # inputs_original[idx] = np.stack(inputs_original[idx], 0)

        output = self.video_readers[current_run_dir]["output_attention"][current_frame_index]

        # start constructing the output dictionary => keep the original frames in there
        # out = {"original": {f"input_stack_{idx}": np.array(i.copy()) for idx, i in enumerate(inputs)}}
        out = {"original": {}}
        # for idx, _ in enumerate(self.input_names):
        #     out["original"][f"input_stack_{idx}"] = np.array(inputs_original[idx])
        out["original"]["output_attention"] = np.array(output)

        # apply transform only for the output
        for idx, i in enumerate(inputs):
            out[f"input_image_{idx}"] = i
        # out["output_attention"] = {k: self.output_transforms[k](output) for k in self.output_transforms}
        # TODO: this is very ugly right now, need to figure out some way to change it
        out["output_attention"] = self.output_transforms["stack"](output)
        if self.dreyeve_transforms:
            out["output_attention_crop"] = self.output_transforms["stack_crop"](output)

        # return a dictionary
        return out


class DrEYEveDataset(StackedImageDataset, ToAttentionDataset):
    # TODO: since the way of dealing with the transforms is pretty annoying, this might be for the best

    def __init__(self, config: dict, split: str, cv_split: int = -1, training: bool = True):
        # TODO: should maybe be set in config and just be asserts here?
        config["dreyeve_transforms"] = True
        config["stack_size"] = 16

        super().__init__(config, split, cv_split, training)

        self.transform = DrEYEveTransform(self.video_input_names)

    def __getitem__(self, item):
        image_stack = self._get_image_stack(item)
        _, attention_original = self._get_attention(item)
        label = self.index["label"].iloc[item]

        # original
        # out = {"original": {f"input_image_{idx}": i for idx, i in enumerate(image_original)}}
        # out = {"original": {"output_attention": attention_original}}
        out = {}

        # to be transformed, which has to happen "outside" the individual "get" methods for dreyeve
        for idx, i in enumerate(image_stack):
            out[f"input_image_{idx}"] = i
        out["output_attention"] = attention_original
        out["label_high_level"] = label

        # transform
        out = self.transform.apply_transforms(out)

        return out


"""
class StackedImageAndStateToControlDataset(Dataset):

    def __init__(self, config, split, cv_split=-1):
        super().__init__()
        self.data_root = config["data_root"]
        self.video_input_names = config["input_video_names"]
        self.state_input_names = config["drone_state_names"]
        self.output_name = config["ground_truth_name"]
        self.output_columns = []
        self.split = split
        self.stack_size = config["stack_size"]
        self.dreyeve_transforms = config["dreyeve_transforms"]

        frame_index = pd.read_csv(os.path.join(self.data_root, "index", "frame_index.csv"))
        split_index = pd.read_csv(config["split_config"] + ".csv")
        if cv_split >= 0:
            split_index = split_index[f"split_{cv_split}"]
        frame_index["split"] = split_index

        ground_truth = pd.read_csv(os.path.join(self.data_root, "index", f"{self.output_name}.csv"))
        for col in ground_truth:
            frame_index[col] = ground_truth[col]
            self.output_columns.append(col)

        drone_state = pd.read_csv(os.path.join(self.data_root, "index", "state.csv"))
        for col in self.state_input_names:
            frame_index[col] = drone_state[col]

        self.index = frame_index[frame_index["split"] == self.split].copy().reset_index(drop=True)

        self.index["label"] = 4
        for idx, (track_name, half) in enumerate([("flat", "left_half"), ("flat", "right_half"),
                                                  ("wave", "left_half"), ("wave", "right_half")]):
            self.index.loc[(self.index["track_name"] == track_name) & (self.index[half] == 1), "label"] = idx

        with open(config["split_config"] + "_info.json", "r") as f:
            split_index_info = json.load(f)
        input_statistics = {}
        for i in self.video_input_names:
            if config["no_normalisation"]:
                input_statistics[i] = {"mean": np.array([0, 0, 0]), "std": np.array([1, 1, 1])}
            else:
                if "mean" in split_index_info and "std" in split_index_info:
                    input_statistics[i] = {
                        "mean": split_index_info["mean"][i] if i in split_index_info["mean"] else STATISTICS["mean"][i],
                        "std": split_index_info["std"][i] if i in split_index_info["std"] else STATISTICS["std"][i]
                    }
                else:
                    input_statistics[i] = {
                        "mean": STATISTICS["mean"][i],
                        "std": STATISTICS["std"][i]
                    }

        self.random_crop = None
        if self.dreyeve_transforms:
            self.random_crop = ManualRandomCrop((256, 256), (112, 112))

            self.video_input_transforms = {
                "stack": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.video_input_names],
                "stack_crop": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    self.random_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"]),
                ]) for i in self.video_input_names],
                "last_frame": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.video_input_names],
            }

            # don't need any output transforms here
        else:
            # just a simple transform for every frame
            self.video_input_transforms = {
                "stack": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(config["resize"]),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.video_input_names]
            }

        self.video_readers = {}

        # find contiguous sequences of frames
        sequences = []
        frames = self.index["frame"]
        jumps = (frames - frames.shift()) != 1
        frames = list(frames.index[jumps]) + [frames.index[-1] + 1]
        for i, start_index in enumerate(frames[:-1]):
            # note that the range [start_frame, end_frame) is exclusive
            sequences.append((start_index, frames[i + 1]))

        # need some way to not mess up the indexing, but also cannot remove data
        # maybe use second "index" that skips the (stack_size - 1) first frames of each contiguous sequence?
        self.index["stack_index"] = -1
        df_col_index = self.index.columns.get_loc("stack_index")
        total_num_frame_stacks = 0
        for start_index, end_index in sequences:
            num_frames_seq = end_index - start_index
            num_frame_stacks = num_frames_seq - self.stack_size + 1
            if end_index - start_index < self.stack_size:
                continue

            # assign index over entire dataframe to "valid" frames
            index = np.arange(num_frame_stacks) + total_num_frame_stacks
            self.index.iloc[(start_index + self.stack_size - 1):end_index, df_col_index] = index

            # keep track of the current number of stacks (highest index)
            total_num_frame_stacks += num_frame_stacks

    def __len__(self):
        return self.index["stack_index"].max() + 1

    def __getitem__(self, item):
        # get the information about the current item
        # TODO: maybe just change the index of the dataframe? might make this particular step easier
        current_stack_index = self.index.index[self.index["stack_index"] == item].values[0]
        # current_run_dir = self.index["run_dir"].iloc[current_stack_index]
        current_stack_df = self.index.iloc[(current_stack_index - self.stack_size + 1):(current_stack_index + 1)]
        current_frame_index = current_stack_df["frame"].iloc[-1]
        current_run_dir = os.path.join(self.data_root, run_info_to_path(current_stack_df["subject"].iloc[0],
                                                                        current_stack_df["run"].iloc[0],
                                                                        current_stack_df["track_name"].iloc[0]))

        # initialise the video readers if necessary
        if current_run_dir not in self.video_readers:
            # print(os.path.join(current_run_dir, f"{self.video_input_names[1]}.mp4"))
            self.video_readers[current_run_dir] = {
                f"input_image_{idx}": get_indexed_reader(os.path.join(current_run_dir, f"{i}.mp4"))
                for idx, i in enumerate(self.video_input_names)
            }

        # prepare new "crop parameters" to get a random crop consistent across different tensors
        if self.random_crop is not None:
            self.random_crop.update()

        # read the frames
        video_inputs = [{k: [] for k in self.video_input_transforms} for _ in self.video_input_names]
        # inputs_original = [{} for _ in self.input_names]
        for idx, _ in enumerate(self.video_input_names):
            for in_stack_idx, (_, row) in enumerate(current_stack_df.iterrows()):
                frame = self.video_readers[current_run_dir][f"input_image_{idx}"][row["frame"]]

                # apply input transform here already
                for k in self.video_input_transforms:
                    if k != "last_frame":
                        video_inputs[idx][k].append(self.video_input_transforms[k][idx](frame))
                    elif in_stack_idx == self.stack_size - 1:
                        video_inputs[idx][k] = self.video_input_transforms[k][idx](frame)

            # stack the frames
            for k in self.video_input_transforms:
                if k != "last_frame":
                    video_inputs[idx][k] = torch.stack(video_inputs[idx][k], 1)

        # read the state input
        state_input = current_stack_df[self.state_input_names].iloc[-1].values

        # determine the "high level command" label for the sample
        high_level_label = current_stack_df["label"].iloc[-1]

        # extract the control GT from the dataframe
        output = current_stack_df[self.output_columns].iloc[-1].values

        # start constructing the output dictionary, left empty for now
        # out = {"original": {}} => strange interaction with get_batch_size if left empty
        out = {}

        # apply transform only for the output
        for idx, i in enumerate(video_inputs):
            out[f"input_image_{idx}"] = i
        out["input_state"] = torch.from_numpy(state_input).float()
        out["output_control"] = torch.from_numpy(output).float()
        out["label_high_level"] = high_level_label

        # return a dictionary
        return out
"""


if __name__ == "__main__":
    test_config = {
        "data_root": os.getenv("GAZESIM_ROOT"),
        "input_video_names": ["screen"],  # ["screen"],
        "drone_state_names": ["DroneVelocityX", "DroneVelocityY", "DroneVelocityZ"],
        "attention_ground_truth": "moving_window_frame_mean_gt",
        "control_ground_truth": "drone_control_frame_mean_gt",
        "resize": 150,
        "stack_size": 8,
        "split_config": resolve_split_index_path(11, data_root=os.getenv("GAZESIM_ROOT")),
        "frames_per_second": 60,
        "no_normalisation": True,
        "video_data_augmentation": False,
        "video_random_cropping": True,
        "vrc_factor_before_crop": 1.5,
        "control_normalisation": True,
        "control_normalisation_range": {
            "throttle": 0.0001,
            "roll": 0.5,
            "pitch": 10.0,
            "yaw": 0.01,
        },
        "clip_gaze": True,
        "scale_gaze": True,
        "dreyeve_transforms": False,
        "feature_track_name": "ft_flightmare_60",
        "feature_track_num": 40,
        "feature_track_sampling_mode": "",
        "reference_name": "drone_state_original",
        "reference_variables": ["rotation_w", "rotation_x", "rotation_y", "rotation_z"],
        "state_estimate_name": "drone_state_original",
        "state_estimate_variables": ["rotation_w", "rotation_x", "rotation_y", "rotation_z"],
        "state_estimate_data_augmentation": False,
    }

    # dataset = DrEYEveDataset(test_config, "train")
    # dataset = ImageToControlDataset(test_config, "train")
    # dataset = StackedImageToControlDataset(test_config, "train")
    # dataset = DDADataset(test_config, "train")
    # dataset = ImageToGazeDataset(test_config, "val")
    dataset = ImageToAttentionDataset(test_config, "train")
    print("dataset size:", len(dataset))

    # from tqdm import tqdm
    # for d in tqdm(dataset):
    #     pass

    sample = dataset[len(dataset) - 1]
    sample = dataset[0]
    print("sample:", sample.keys())
    # print(sample["output_gaze"])
    # print(sample["output_control"])
    # print(sample["input_feature_tracks"].shape)
    # print(sample["input_feature_tracks"]["stack"].shape)
    # print(sample["input_state"]["stack"].shape)
    # print(sample["input_image_0"]["stack"].shape)
    # print(sample["output_control"])
    # print(sample["input_image_0"].keys())
    # for k, v in sample["input_image_0"].items():
    #     print(v.shape)
    # print(sample["input_state"].shape)
    # print(sample["output_attention"].shape)
    # print(sample["output_control"].shape)

    exit(0)

    test_config = {
        "data_root": os.getenv("GAZESIM_ROOT"),
        "input_video_names": ["screen"],
        # "ground_truth_name": "moving_window_frame_mean_gt",
        "ground_truth_name": "drone_control_frame_mean_gt",
        "drone_state_names": ["DroneVelocityX", "DroneVelocityY", "DroneVelocityZ"],
        "resize": (122, 122),
        "split_config": resolve_split_index_path(11, data_root=os.getenv("GAZESIM_ROOT")),
        "stack_size": 4,
        "dreyeve_transforms": False,
        "no_normalisation": False
    }

    dataset = StackedImageAndStateToControlDataset(test_config, split="val")
    print(len(dataset))
    # print(dataset.index.columns)

    sample = dataset[0]
    print("sample:", sample.keys())

    exit(0)

    print("\ninput_image_0:", sample["input_image_0"].keys())
    for k, v in sample["input_image_0"].items():
        print(f"{k}: {v.shape}")

    # print("\ninput_image_1:", sample["input_image_1"].keys())
    # for k, v in sample["input_image_1"].items():
    #     print(f"{k}: {v.shape}")

    print("\noutput_attention:", sample["output_attention"].keys())
    for k, v in sample["output_attention"].items():
        print(f"{k}: {v.shape}")
