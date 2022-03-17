import fnmatch
import os
import random
import cv2

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.transform import Rotation as R


def create_dataset(directory, settings, training=True):
    dataset = SafeDataset(directory, settings, training)
    return dataset


class BodyDataset:
    """
    Base Dataset Class
    """

    def __init__(self, directory, config, training=True):
        self.config = config
        self.directory = directory
        self.training = training
        self.samples = 0
        self.experiments = []
        self.features = []
        self.labels = []
        self.attention_labels = []
        self.gate_direction_labels = []
        self.filenames = []
        self.stacked_filenames = []  # Will be used for passing stacked fnames
        self.att_fts_filenames = []
        img_rootname = "img_data"
        att_fts_rootname = self.config.attention_fts_type
        att_fts_experiments = []
        for root, dirs, files in os.walk(directory, topdown=True, followlinks=True):
            for name in dirs:
                if name.startswith(img_rootname):
                    exp_dir = os.path.join(root, name)
                    self.experiments.append(os.path.abspath(exp_dir))
                elif att_fts_rootname != "none" and name.startswith(att_fts_rootname):
                    exp_dir = os.path.join(root, name)
                    att_fts_experiments.append(os.path.abspath(exp_dir))

        if self.config.attention_fts_type != "none":
            self.experiments = sorted(self.experiments)
            att_fts_experiments = sorted(att_fts_experiments)
            self.experiments = list(zip(self.experiments, att_fts_experiments))

            # check that time stamps/names match
            if not all([exp.endswith(af_exp[-15:]) for exp, af_exp in self.experiments]):
                raise ImportError("With attention feature type '{}' specified, feature track and attention "
                                  "feature folders do not match".format(self.config.attention_fts_type))

        # TODO: need to match this with features....
        # one other "issue": if different experiments have different feature types everything
        # falls apart, but that would be really dumb and more of a user error

        self.num_experiments = len(self.experiments)
        self.img_format = "npy"
        if self.config.use_images:
            self.img_format = "jpg"
        self.data_format = "csv"
        self.att_fts_format = "npy"

        for exp_dir in self.experiments:
            try:
                self._decode_experiment_dir(exp_dir)
            except:
                raise ImportWarning("Image reading in {} failed".format(exp_dir))
        if self.samples == 0:
            raise IOError("Did not find any file in the dataset folder")
        print("[BodyDataset] Found {} images belonging to {} experiments:".format(self.samples, self.num_experiments))

    def _recursive_list(self, subpath, fmt="jpg"):
        return fnmatch.filter(os.listdir(subpath), "*.{}".format(fmt))

    def build_dataset(self):
        self._build_dataset()

    def _build_dataset(self):
        raise NotImplementedError

    def _decode_experiment_dir(self, directory):
        raise NotImplementedError


class SafeDataset(BodyDataset):

    def __init__(self, directory, config, training=True):
        super(SafeDataset, self).__init__(directory, config, training)
        self.build_dataset()

    def _decode_experiment_dir(self, dir_subpath):
        if not isinstance(dir_subpath, tuple):
            dir_subpath = (dir_subpath,)

        base_path = os.path.basename(dir_subpath[0])
        parent_dict = os.path.dirname(dir_subpath[0])

        # load the state/command data
        data_name = "data" + base_path[8:] + ".csv"
        data_name = os.path.join(parent_dict, data_name)
        assert os.path.isfile(data_name), "Not Found data file"
        df = pd.read_csv(data_name, delimiter=",")
        num_files = df.shape[0]

        # get the number of saved feature track dicts TODO: do the same for attention fts (also assert stuff)
        num_images = len(self._recursive_list(dir_subpath[0], fmt=self.img_format))
        if self.config.attention_fts_type != "none":
            num_att_fts = len(self._recursive_list(dir_subpath[1], fmt=self.att_fts_format))
            assert num_files == num_images == num_att_fts, \
                "Number of state measurements, images and attention features does not match"
        else:
            assert num_files == num_images, "Number of state measurements and images does not match"

        features_imu = [  # VIO Estimate
            "Orientation_x",
            "Orientation_y",
            "Orientation_z",
            "Orientation_w",
            "V_linear_x",
            "V_linear_y",
            "V_linear_z",
            "V_angular_x",
            "V_angular_y",
            "V_angular_z"]
        # the above should maybe also be replaced by the GT (maybe w/ some noise)
        if self.config.use_pos:
            features_imu += [
                "Position_x",
                "Position_y",
                "Position_z",
            ]

        # TODO: this probably needs to be changed to the available reference states
        #  (i.e. position, linear velocity, rotation, which are probably more relevant
        #  for racing stuff) => will probably want to keep body rates
        # TODO: if this is changed (at least) preprocess_fts also needs to be changed
        features = [  # Reference state
            "Reference_orientation_x",
            "Reference_orientation_y",
            "Reference_orientation_z",
            "Reference_orientation_w",
            "Reference_v_linear_x",
            "Reference_v_linear_y",
            "Reference_v_linear_z",
            "Reference_v_angular_x",
            "Reference_v_angular_y",
            "Reference_v_angular_z"]
        if self.config.use_pos:
            features += [
                "Reference_position_x",
                "Reference_position_y",
                "Reference_position_z",
            ]

        # Preprocessing: we select the good rollouts (no crash in training data)
        rollout_fts = ["Rollout_idx"]
        rollout_fts_v = df[rollout_fts].values
        position_gt = ["gt_Position_x",
                       "gt_Position_y",
                       "gt_Position_z"]
        position_gt_v = df[position_gt].values
        position_ref = ["Reference_position_x",
                        "Reference_position_y",
                        "Reference_position_z"]
        position_ref_v = df[position_ref].values
        collision_v = df["Collision"].values

        good_rollouts = []

        if rollout_fts_v.shape[0] == 0:
            return

        for r in np.arange(1, np.max(rollout_fts_v) + 1):
            rollout_positions = rollout_fts_v == r
            roll_gt = position_gt_v[np.squeeze(rollout_positions), :]
            roll_ref = position_ref_v[np.squeeze(rollout_positions), :]
            roll_collision = collision_v[np.squeeze(rollout_positions)]

            if roll_gt.shape[0] == 0:
                continue
            if self.config.exclude_collision_rollouts and np.sum(roll_collision) > 0:
                print(f"COLLISION ROLLOUT: {r}")
                continue

            assert roll_ref.shape == roll_gt.shape
            error = np.mean(np.linalg.norm(roll_gt - roll_ref, axis=1))
            if error < self.config.max_allowed_error:
                good_rollouts.append(r)

        if self.config.use_imu:
            features = ["Rollout_idx"] + features_imu + features
        else:
            features = ["Rollout_idx"] + features

        # TODO: high-level label (from gaze/velocity vector comparison) should probably just be added to the
        #  features here and in the map function, it should just take the last "column" of the features?

        labels = ["Gt_control_command_collective_thrust",
                  "Gt_control_command_bodyrates_x",
                  "Gt_control_command_bodyrates_y",
                  "Gt_control_command_bodyrates_z"]

        attention_label = "Attention_label"
        gate_direction_label = "Gate_direction_label"

        features_v = df[features].values
        labels_v = df[labels].values

        attention_label_v = None
        gate_direction_label_v = None
        if self.config.attention_branching:
            attention_label_v = df[attention_label].values
        elif self.config.gate_direction_branching:
            gate_direction_label_v = df[gate_direction_label].values

        for frame_number in range(num_files):
            is_valid = False
            img_fname = os.path.join(dir_subpath[0], "{:08d}.{}".format(frame_number, self.img_format))
            att_fts_fname = None
            if self.config.attention_fts_type != "none":
                att_fts_fname = os.path.join(dir_subpath[1], "{:08d}.{}".format(frame_number, self.att_fts_format))
            if os.path.isfile(img_fname) \
                    and (self.config.attention_fts_type == "none" or os.path.isfile(att_fts_fname)) \
                    and (rollout_fts_v[frame_number] in good_rollouts):
                is_valid = True
            if is_valid:
                self.features.append(self.preprocess_fts(features_v[frame_number]))
                self.labels.append(labels_v[frame_number])
                if self.config.use_fts_tracks or self.config.use_images:
                    self.filenames.append(img_fname)
                if self.config.attention_fts_type != "none":
                    self.att_fts_filenames.append(att_fts_fname)
                if self.config.attention_branching:
                    self.attention_labels.append(attention_label_v[frame_number])
                elif self.config.gate_direction_branching:
                    self.gate_direction_labels.append(gate_direction_label_v[frame_number])
                self.samples += 1

    def preprocess_fts(self, fts):
        """
        Converts rotations from quadrans to rotation matrix.
        Fts have the following indexing.
        rollout_idx, qx,qy,qz,qw, vx, vy, vz, ax, ay, az, rqx, rqy, rqz, rqw, ...
        """
        fts = fts.tolist()
        # I don't think these indices are correct... they only work if IMU data is also used
        # ref_rot = R.from_quat(fts[11:15]).as_matrix().reshape((9,)).tolist()
        if self.config.use_imu:
            if self.config.use_pos:
                ref_rot = R.from_quat(fts[14:18]).as_matrix().reshape((9,)).tolist()
            else:
                ref_rot = R.from_quat(fts[11:15]).as_matrix().reshape((9,)).tolist()

            odom_rot = R.from_quat(fts[1:5]).as_matrix().reshape((9,)).tolist()
            if self.config.use_pos:
                processed_fts = [fts[0]] + ([] if self.config.imu_no_rot else odom_rot) + \
                                ([] if self.config.imu_no_vels else fts[5:14])
                if not self.config.no_ref:
                    processed_fts += ref_rot + fts[18:]
            else:
                processed_fts = [fts[0]] + ([] if self.config.imu_no_rot else odom_rot) + \
                                ([] if self.config.imu_no_vels else fts[5:11])
                if not self.config.no_ref:
                    processed_fts += ref_rot + fts[15:]
        else:
            ref_rot = R.from_quat(fts[1:5]).as_matrix().reshape((9,)).tolist()
            if self.config.use_pos:
                processed_fts = [fts[0]]
                if not self.config.no_ref:
                    processed_fts += ref_rot + fts[5:14]
            else:
                processed_fts = [fts[0]]
                if not self.config.no_ref:
                    processed_fts += ref_rot + fts[5:11]
        return np.array(processed_fts)

    def add_missing_fts(self, features_dict):
        processed_dict = features_dict
        # Could be both positive or negative
        missing_fts = self.config.min_number_fts - len(features_dict.keys())
        if missing_fts > 0:
            # Features are missing
            if missing_fts != self.config.min_number_fts:
                # There is something, we can sample
                new_features_keys = random.choices(list(features_dict.keys()), k=int(missing_fts))
                for j in range(missing_fts):
                    processed_dict[-j - 1] = features_dict[new_features_keys[j]]
            else:
                # Zero features, this is a transient
                for j in range(missing_fts):
                    processed_dict[-j - 1] = np.zeros((5,))
        elif missing_fts < 0:
            # There are more features than we need, so sample
            del_features_keys = random.sample(features_dict.keys(), int(-missing_fts))
            for k in del_features_keys:
                del processed_dict[k]
        return processed_dict

    def load_fts_sequence(self, sample_num):
        fts_seq = []
        sample_num_np = sample_num.numpy()
        filenames_num = self.stacked_filenames[sample_num_np]
        for idx in range(self.config.seq_len):
            fname_idx = filenames_num[idx]
            if fname_idx < 0:
                fts = {}
            else:
                fname = self.filenames[fname_idx]
                fts = np.load(fname, allow_pickle=True).item()
            fts_seq.append(fts)
        # Reverse list to have it ordered in time (t-seq_len, ..., t)
        fts_seq = reversed(fts_seq)
        # Crop to the required lenght
        fts_seq = [self.add_missing_fts(ft) for ft in fts_seq]
        # Stack
        features_input = np.stack([np.stack([v for v in fts_seq[j].values()]) \
                                   for j in range(self.config.seq_len)])
        return features_input

    def load_att_fts_sequence(self, sample_num):
        att_fts_seq = np.load(self.att_fts_filenames[sample_num.numpy()])
        if att_fts_seq.shape[0] > self.config.seq_len:
            att_fts_seq = att_fts_seq[-self.config.seq_len:]
        elif att_fts_seq.shape[0] < self.config.seq_len:
            raise ImportError("Attention features can only be loaded with seq_len <= what they were saved with.")
        return att_fts_seq

    def _dataset_map(self, sample_num):
        # first is rollout idx
        inputs = []
        label = tf.gather(self.labels, sample_num)

        # for states it is easy: nothing to do
        state_seq = []
        for idx in reversed(range(self.config.seq_len)):
            state = tf.gather(self.features, sample_num - idx)[1:]
            state_seq.append(state)
        state_seq = tf.stack(state_seq)
        inputs.append(state_seq)

        # TODO: if config.no_ref and not config.use_imu, no state input should be returned
        #  => maybe it's easier to just ignore this when learning instead...

        # TODO: if decision variable for branching specified, gather here
        #  => might be better not to leave it in self.features, since it becomes part of the state then

        # for images, take care they do not overlap
        if self.config.use_fts_tracks:
            fts_seq = tf.py_function(func=self.load_fts_sequence, inp=[sample_num], Tout=tf.float32)
            inputs.append(fts_seq)
        elif self.config.use_images:
            image_stack = tf.py_function(func=self.load_img_sequence, inp=[sample_num], Tout=tf.uint8)
            image_stack = self._preprocess_img(image_stack)
            inputs.append(image_stack)

        # for attention features all features are saved in the file for now, so it just has to be loaded
        if self.config.attention_fts_type != "none":
            att_fts_seq = tf.py_function(func=self.load_att_fts_sequence, inp=[sample_num], Tout=tf.float32)
            inputs.append(att_fts_seq)

        if self.config.attention_branching:
            att_label = tf.gather(self.attention_labels, sample_num)
            inputs.append(att_label)
        elif self.config.gate_direction_branching:
            gate_dir_label = tf.gather(self.gate_direction_labels, sample_num)
            inputs.append(gate_dir_label)

        """
            return (state_seq, fts_seq), label
        else:
            return state_seq, label#
        """

        inputs = tuple(inputs)
        return inputs, label

    def check_equal_dict(self, d1, d2):
        for k_1, v_1 in d1.items():
            try:
                v_2 = d2[k_1]
            except:
                return False
            if not np.array_equal(v_1, v_2):
                return False
        return True

    def _preprocess_fnames(self):
        # TODO: this is still the biggest mystery
        #  => need to figure out what this whole "overlapping" is supposed to mean
        #  => does it mean that only features that are not the same are used as input?
        #     that seems like it doesn't make sense since features might indeed still be the same, no?
        #  => wait, nevermind, the whole point of the asynchronous architecture is that the features can be recorded
        #     with slower frequency than e.g. the states, but since everything is saved at the "network" command
        #     frequency some of the recorded feature files should actually be skipped!!

        # Append filenames up to seq_len for fast loading.
        # A bit ugly and inefficient, can be improved
        self.last_init_fts = None
        check_func = np.array_equal if self.config.use_images else self.check_equal_dict
        for k in range(len(self.filenames)):
            if k % 3000 == 0:
                print("[BodyDataset] Built {:.2f}% of the dataset".format(k / len(self.filenames) * 100), end="\r")
            # Check if you can copy the things before
            # TODO: I guess this is relevant because the frequency is low enough that upon save the same feature
            #  tracks might be written to files multiple times
            #  => thus, just (pre-)loading one file is more efficient
            #  => the same can be done (even easier) with attention fts, just check array equality
            kth_fts = self.filenames[k]
            if self.config.use_images:
                kth_fts = cv2.imread(kth_fts)
            else:
                kth_fts = np.load(kth_fts, allow_pickle=True).item()
            if k > 0:
                if check_func(self.last_init_fts, kth_fts):
                    self.stacked_filenames.append(self.stacked_filenames[-1])
                    continue

            # This is the latest observed feature track different from others
            self.last_init_fts = kth_fts
            idx = 0
            rollout_indices = []
            fname_seq = []
            fts_seq = []
            while len(fts_seq) < self.config.seq_len:
                if k - idx < 0:
                    # this is transient, can only append zeros
                    fname_seq.append(-1)
                    fts_seq.append(0.)
                    continue
                current_idx = k - idx
                rollout_idx = self.features[current_idx][0]
                if idx == 0:
                    fname_seq.append(current_idx)
                    fts_seq.append(kth_fts)
                    rollout_indices.append(rollout_idx)
                else:
                    if rollout_idx != rollout_indices[-1]:
                        # it is a transient! Can only append zeros.  TODO: I guess they mean transition?
                        fname_seq.append(-1)
                        fts_seq.append(0.)
                    else:
                        # Check the features are different
                        fname = self.filenames[current_idx]
                        fts = cv2.imread(fname) if self.config.use_images else np.load(fname, allow_pickle=True).item()
                        if not check_func(fts, fts_seq[-1]):
                            # Objects are not equal, can append
                            fname_seq.append(current_idx)
                            fts_seq.append(fts)
                            rollout_indices.append(rollout_idx)
                        # TODO: so I think this basically just tries to go back through the last x features...
                        #  no wtf I have no clue what this is doing
                idx += 1
            assert len(fts_seq) == len(fname_seq)
            self.stacked_filenames.append(fname_seq)
        # EndFor
        assert len(self.filenames) == len(self.stacked_filenames)

    def _preprocess_img(self, image):
        image = tf.cast(image, dtype=tf.float32)
        image = 2 * (image / 255. - 0.5)
        return image

    def load_img_sequence(self, sample_num):
        image_seq = []
        sample_num_np = sample_num.numpy()
        filenames_num = self.stacked_filenames[sample_num_np]
        for idx in range(self.config.seq_len):
            fname_idx = filenames_num[idx]
            if fname_idx < 0:
                image = np.zeros((300, 400, 3))
            else:
                fname = self.filenames[fname_idx]
                image = cv2.imread(fname)
            image_seq.append(image)

        # Reverse list to have it ordered in time (t-seq_len, ..., t)
        image_seq = reversed(image_seq)
        image_seq = np.stack(image_seq)
        return image_seq

    def _build_dataset(self):
        # Need to take care that rollout_idxs are consistent
        self.features = np.stack(self.features)
        self.features = self.features.astype(np.float32)
        self.labels = np.stack(self.labels)
        self.labels = self.labels.astype(np.float32)
        last_fname_numbers = []
        if self.config.use_fts_tracks or self.config.use_images:
            self._preprocess_fnames()
        # Preprocess filenames to assess consistency of experiment
        for idx in range(self.config.seq_len - 1, self.samples):
            if self.features[idx, 0] == self.features[idx - self.config.seq_len + 1, 0]:
                # so I guess this includes all the file name "indices" while we are still in the same rollout?
                last_fname_numbers.append(np.int32(idx))

        if self.training:
            np.random.shuffle(last_fname_numbers)

        # Form training batches
        dataset = tf.data.Dataset.from_tensor_slices(last_fname_numbers)
        if self.training:
            dataset = dataset.shuffle(buffer_size=len(last_fname_numbers))
        dataset = dataset.map(self._dataset_map, num_parallel_calls=10 if self.training else 1)  # applies the function to each element of the dataset
        dataset = dataset.batch(self.config.batch_size, drop_remainder=not self.training)  # guess this is just to specify batch size and other behaviour
        dataset = dataset.prefetch(buffer_size=10 * self.config.batch_size)
        self.batched_dataset = dataset


if __name__ == "__main__":
    from dda.config.settings import create_settings
    from pprint import pprint

    load_dir = "/home/simon/gazesim-data/fpv_saliency_maps/data/dda/data/testing/train/"
    settings_path = "../config/dagger_settings.yaml"
    settings = create_settings(settings_path, mode="dagger")
    settings.batch_size = 1
    ds = create_dataset(load_dir, settings)

    # pprint(ds.stacked_filenames)
    # print()
    # pprint(ds.filenames)

    c = 0
    for k, (features, label) in enumerate(ds.batched_dataset):
        if c > 5:
            break

        state_seq, fts_seq, att_fts_seq = features
        print(state_seq.shape)
        print(fts_seq.shape)
        print(att_fts_seq.shape)
        print("\n\n\n")

        c += 1
