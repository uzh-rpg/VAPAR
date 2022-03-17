import numpy as np
import random
import copy
import os
import csv
import datetime
import cv2
import warnings

from collections import deque
from scipy.spatial.transform import Rotation
from dda.models.bodyrate_learner import BodyrateLearner
from features.feature_tracker import FeatureTracker
from features.attention import AttentionEncoderFeatures, AttentionMapTracks, GazeTracks
from features.attention import AllAttentionFeatures, AttentionHighLevelLabel, AttentionMasking
from features.gates import GateDirectionHighLevelLabel
from planning.mpc_solver import MPCSolver
from planning.planner import TrajectoryPlanner

# TODO: not sure why this is the way it is, might have to be adjusted
# e.g. if it stems from most features in the original only being tracked for around 10 steps
# then this normalisation won't do much to bring the value into a similar range as the other
# components of the feature track "features"
TRACK_NUM_NORMALIZE = 10


class ControllerLearning:

    def __init__(self, config, trajectory_path, mode, max_time=None):
        # "meta" stuff
        self.config = config
        self.mode = mode

        self.csv_filename = None
        self.image_save_dir = None
        self.attention_fts_save_dir = None

        # things to keep track of the current "status"
        self.record_data = False
        self.record_test_data = False
        self.is_training = False
        self.use_network = False
        self.network_initialised = False
        self.reference_updated = False  # this just seems to be a flag to make sure that there is a reference state at all

        self.rollout_idx = 0
        self.n_times_net = 0
        self.n_times_expert = 0
        self.n_times_randomised = 0
        self.recorded_samples = 0
        self.counter = 0

        self.state_queue = deque([], maxlen=self.config.seq_len)
        self.fts_queue = deque([], maxlen=self.config.seq_len)
        self.attention_fts_queue = deque([], maxlen=self.config.seq_len)
        self.image_queue = deque([], maxlen=self.config.seq_len)

        # simulation time (for now mostly for the feature tracker for velocity calculation)
        self.simulation_time = 0.0

        # stuff to keep track of
        self.state = None
        self.state_rot = None
        self.reference = None
        self.reference_rot = None
        self.state_estimate = None
        self.state_estimate_rot = None
        self.feature_tracks = None
        self.image = None
        self.attention_label = 0
        self.gate_direction_label = 0
        self.collision = 0

        self.extra_info = {}

        # the current control command, computed either by the expert or the network
        self.control_command = None
        self.network_command = None

        # objects
        self.feature_tracker = FeatureTracker(int(self.config.min_number_fts * 1.5))
        self.learner = BodyrateLearner(settings=self.config, expect_partial=(mode == "testing"))
        self.planner = TrajectoryPlanner(
            trajectory_path=trajectory_path,
            plan_time_horizon=self.config.mpc_time_horizon,
            plan_time_step=self.config.mpc_time_step,
            max_time=max_time,
        )
        self.expert = MPCSolver(self.config.mpc_time_horizon, self.config.mpc_time_step)

        # attention features
        self.attention_fts_extractor = None
        self.attention_fts_size = -1
        if self.config.attention_record_all_features:
            self.attention_fts_extractor = AllAttentionFeatures(self.config)
        if self.config.attention_fts_type == "encoder_fts":
            self.attention_fts_size = 475  # 128
            self.attention_fts_extractor = self.attention_fts_extractor or AttentionEncoderFeatures(self.config)
        elif self.config.attention_fts_type == "decoder_fts":
            self.attention_fts_size = 128
            self.attention_fts_extractor = self.attention_fts_extractor or AttentionEncoderFeatures(self.config)
        elif self.config.attention_fts_type == "map_tracks":
            self.attention_fts_size = 4
            self.attention_fts_extractor = self.attention_fts_extractor or AttentionMapTracks(self.config)
        elif self.config.attention_fts_type == "gaze_tracks":
            self.attention_fts_size = 4
            self.attention_fts_extractor = self.attention_fts_extractor or GazeTracks(self.config)

        self.attention_high_level_label_extractor = None
        if self.config.attention_branching:
            self.attention_high_level_label_extractor = AttentionHighLevelLabel(self.config)

        self.attention_masker = None
        self.gate_direction_high_level_label_extractor = None
        if self.config.attention_masking:
            self.attention_masker = AttentionMasking(self.config)
        elif self.config.gate_direction_branching:
            self.gate_direction_high_level_label_extractor = GateDirectionHighLevelLabel(self.config)

        # preparing for data saving
        """
        if self.mode == "iterative" or self.config.verbose:
            self.write_csv_header()
        """

    def reset(self, new_rollout=True):
        if new_rollout:
            self.rollout_idx += 1
        self.n_times_net = 0
        self.n_times_expert = 0
        self.n_times_randomised = 0
        self.counter = 0

        self.use_network = True

        self.fts_queue.clear()
        self.state_queue.clear()
        self.attention_fts_queue.clear()

        self.state = np.zeros((13,), dtype=np.float32)
        self.reference = np.zeros((13,), dtype=np.float32)
        self.state_estimate = np.zeros((13,), dtype=np.float32)
        self.attention_label = 0
        self.collision = 0

        self.extra_info = {}

        if self.config.use_imu:
            if self.config.use_pos:
                n_init_states = 36
            else:
                n_init_states = 30
            if self.config.imu_no_rot:
                n_init_states -= 9
            if self.config.imu_no_vels:
                n_init_states -= 6
            if self.config.no_ref:
                n_init_states -= 18 if self.config.use_pos else 15
        else:
            if self.config.use_pos:
                n_init_states = 18
            else:
                n_init_states = 15
            if self.config.no_ref:
                n_init_states = 0

        init_dict = {}
        for i in range(self.config.min_number_fts):
            init_dict[i] = np.zeros((5,), dtype=np.float32)

        for _ in range(self.config.seq_len):
            self.state_queue.append(np.zeros((n_init_states,), dtype=np.float32))
            self.fts_queue.append(init_dict)
            if self.config.attention_record_all_features:
                self.attention_fts_queue.append({att_f_t: np.zeros((att_f_sz,), dtype=np.float32)
                                                 for att_f_t, att_f_sz in
                                                 [("encoder_fts", 475), ("map_tracks", 4), ("gaze_tracks", 4)]})
            elif self.config.attention_fts_type != "none":
                self.attention_fts_queue.append(np.zeros((self.attention_fts_size,), dtype=np.float32))
            if self.config.use_images:
                self.image_queue.append(np.zeros((300, 400, 3), dtype=np.uint8))  # TODO: MIGHT BE DIFFERENT!

        self.feature_tracks = copy.copy(init_dict)
        self.feature_tracker.reset()
        if self.config.gate_direction_branching:
            self.gate_direction_high_level_label_extractor.reset()
        # TODO: also reset attention feature extractor if necessary

    def prepare_data_recording(self):
        if self.mode == "iterative":
            print("\n[ControllerLearning] Directories set and header written\n")
            self.csv_filename = None
            self.image_save_dir = None
            self.attention_fts_save_dir = None
            self.write_csv_header()

    def start_data_recording(self):
        print("\n[ControllerLearning] Collecting data\n")
        self.record_data = True

    def stop_data_recording(self):
        print("\n[ControllerLearning] Stop data collection\n")
        self.record_data = False
        total = self.n_times_net + self.n_times_expert + self.n_times_randomised
        usage = {
            "expert": self.n_times_expert / total if total != 0 else np.nan,
            "network": self.n_times_net / total if total != 0 else np.nan,
            "randomised": self.n_times_randomised / total if total != 0 else np.nan,
        }
        return usage

    def train(self):
        # not sure whether all these booleans are actually relevant
        self.is_training = True
        self.learner.train()
        self.is_training = False
        self.use_network = False

    def update_trajectory(self, trajectory_path, max_time=None, max_time_from_new=False):
        if self.mode != "testing":
            warnings.warn("Trajectory changed/updated during mode '{}'. This functionality is mostly "
                          "intended for testing, but can be used outside outside of that.".format(self.mode))

        print("\n[ControllerLearning] Trajectory updated (should probably reset)\n")

        if max_time_from_new:
            max_time = None
        elif max_time is None:
            max_time = self.planner.get_final_time_stamp()

        self.planner = TrajectoryPlanner(
            trajectory_path=trajectory_path,
            plan_time_horizon=self.config.mpc_time_horizon,
            plan_time_step=self.config.mpc_time_step,
            max_time=max_time,
        )

    def update_simulation_time(self, simulation_time):
        self.simulation_time = simulation_time

    def update_state(self, state):
        # assumed ordering of state variables is [pos. rot, vel, omega]
        # simulation returns full state (with linear acc and motor torques) => take only first 13 entries
        self.state = state[:13]
        self.state_rot = self.state[4:7].tolist() + self.state[3:4].tolist()
        self.state_rot = Rotation.from_quat(self.state_rot).as_matrix().reshape((9,)).tolist()

        if self.config.gate_direction_branching:
            self.gate_direction_label = self.gate_direction_high_level_label_extractor.get_label(
                self.state, self.simulation_time)
            if self.config.return_extra_info:
                self.extra_info.update(self.gate_direction_label[1])
                self.gate_direction_label = self.gate_direction_label[0]

    def update_reference(self, reference):
        self.reference = reference
        self.reference_rot = self.reference[4:7].tolist() + self.reference[3:4].tolist()
        self.reference_rot = Rotation.from_quat(self.reference_rot).as_matrix().reshape((9,)).tolist()
        if not self.reference_updated:
            self.reference_updated = True

    def update_state_estimate(self, state_estimate):
        self.state_estimate = state_estimate[:13]
        self.state_estimate_rot = self.state_estimate[4:7].tolist() + self.state_estimate[3:4].tolist()
        self.state_estimate_rot = Rotation.from_quat(self.state_estimate_rot).as_matrix().reshape((9,)).tolist()

    def update_image(self, image):
        # get the features for the current frame
        feature_tracks = None
        if self.config.use_fts_tracks or self.mode != "testing":
            feature_tracks = self.feature_tracker.process_image(image, current_time=self.simulation_time)

        if feature_tracks is not None:
            # "format" the features like original DDA
            features_dict = {}
            for i in range(len(feature_tracks)):
                ft_id = feature_tracks[i][0]
                x = feature_tracks[i][2]
                y = feature_tracks[i][3]
                velocity_x = feature_tracks[i][4]
                velocity_y = feature_tracks[i][5]
                track_count = 2 * (feature_tracks[i][1] / TRACK_NUM_NORMALIZE) - 1  # TODO: probably revise
                feat = np.array([x, y, velocity_x, velocity_y, track_count])
                features_dict[ft_id] = feat

            if len(features_dict.keys()) != 0:
                # remember the "unsampled" features for saving them for training
                self.feature_tracks = copy.copy(features_dict)

                # sample features
                processed_dict = copy.copy(features_dict)
                missing_fts = self.config.min_number_fts - len(features_dict.keys())
                if missing_fts > 0:
                    # features are missing
                    if missing_fts != self.config.min_number_fts:
                        # there is something, we can sample
                        new_features_keys = random.choices(list(features_dict.keys()), k=int(missing_fts))
                        for j in range(missing_fts):
                            processed_dict[-j - 1] = features_dict[new_features_keys[j]]
                    else:
                        raise IOError("There should not be zero features!")
                elif missing_fts < 0:
                    # there are more features than we need, so sample
                    del_features_keys = random.sample(features_dict.keys(), int(-missing_fts))
                    for k in del_features_keys:
                        del processed_dict[k]

                self.fts_queue.append(processed_dict)

        if self.config.attention_fts_type != "none" or self.config.attention_record_all_features:
            # line(s) below left in in case attention should be needed as output again
            # attention_fts, out_attention = self.attention_fts_extractor.get_attention_features(
            #     image, current_time=self.simulation_time)
            # self.extra_info.update({"out_attention": out_attention})
            attention_fts = self.attention_fts_extractor.get_attention_features(
                image, current_time=self.simulation_time)
            self.attention_fts_queue.append(attention_fts)

        if self.config.attention_branching:
            self.attention_label = self.attention_high_level_label_extractor.get_attention_features(
                image, drone_state=self.state_estimate)
            if self.config.return_extra_info:
                self.extra_info.update(self.attention_label[1])
                self.attention_label = self.attention_label[0]

        if self.config.use_images:
            if self.config.attention_masking:
                image = self.attention_masker.get_masked_image(image)
            self.image = cv2.resize(copy.copy(image), (400, 300))

            # change formatting of image to prepare for input to TensorFlow
            processed_image = 2 * (self.image / 255. - 0.5)
            processed_image = np.array(processed_image, dtype=np.float32)
            self.image_queue.append(processed_image)

    def update_info(self, info_dict):
        if info_dict["collision"]:
            self.collision = 1
        self.update_simulation_time(info_dict["time"])
        self.update_state(info_dict["state"])
        self.update_state_estimate(info_dict["state_estimate"])
        if info_dict["update"]["image"]:
            self.update_image(info_dict["image"])
        if info_dict["update"]["reference"]:
            self.update_reference(info_dict["reference"])
        if info_dict["update"]["command"]:
            self.prepare_network_command()
        if info_dict["update"]["expert"]:
            self.prepare_expert_command()

    def prepare_network_command(self):
        # format the inputs
        inputs = self._prepare_net_inputs()

        # "initialise" the network structure if it hasn't been done already
        if not self.network_initialised:
            results = self.learner.inference(inputs)
            self.network_command = np.array([results[0][0], results[0][1], results[0][2], results[0][3]])
            print("[ControllerLearning] Network initialized")
            self.network_initialised = True
            return

        # print("\nNetwork inputs:")
        # pprint(inputs)
        # print()

        # apply network
        results = self.learner.inference(inputs)
        self.network_command = np.array([results[0][0], results[0][1], results[0][2], results[0][3]])

        # print("Network prediction command: {}, label: {}".format(list(self.network_command), self.attention_label))

    def prepare_expert_command(self):
        # get the reference trajectory over the time horizon
        planned_traj = self.planner.plan(self.state[:10], self.simulation_time)
        planned_traj = np.array(planned_traj)

        # run non-linear model predictive control
        optimal_action, predicted_traj, cost = self.expert.solve(planned_traj)
        self.control_command = optimal_action

    def get_control_command(self):
        control_command_dict = self._generate_control_command()
        return control_command_dict

    def _generate_control_command(self):
        # return self.control_command
        control_command_dict = {
            "expert": self.control_command,
            "network": np.array([np.nan, np.nan, np.nan, np.nan])
            if self.network_command is None else self.network_command,
            "use_network": False,
        }

        # always use expert at the beginning (approximately 0.2s) to avoid synchronization problems
        # => should be a better way of doing this than this counter...
        if self.counter < 10:
            self.counter += 1
            if self.record_data:
                self.n_times_expert += 1
            return control_command_dict

        # TODO: Part of the problem with having "blocking execution" and all that jazz is that the expert
        #  takes much longer to produce commands than the network. In the original DDA this is no problem since
        #  it runs completely independently in the ROS framework. However, trying to compute new expert commands
        #  at the ("targeted") network rate of 100Hz would probably take very long in practice (e.g. running sim.py
        #  with a command frequency of 100Hz). Since it doesn't really make sense to run the expert in parallel,
        #  since the simulation isn't real time and the results would therefore be skewed, one possibility would be
        #  to have two "command frequencies", one for the expert and one for the network, with the former being a lot
        #  lower (e.g. 20Hz) => this command could be manually updated at this lower frequency (e.g. the step method
        #  in PythonSimulation could return a boolean "notifying" this) and then, whenever the expert output should be
        #  used, the "cached" command is used!

        # log everything at the base frequency (i.e. the state "estimate" frequency)
        if self.record_data:
            self.save_data()

        # apply random controller now and then to facilitate exploration
        if (self.mode != "testing") and random.random() < self.config.rand_controller_prob:
            self.control_command[0] += self.config.rand_thrust_mag * (random.random() - 0.5) * 2
            self.control_command[1] += self.config.rand_rate_mag * (random.random() - 0.5) * 2
            self.control_command[2] += self.config.rand_rate_mag * (random.random() - 0.5) * 2
            self.control_command[3] += self.config.rand_rate_mag * (random.random() - 0.5) * 2
            # TODO: maybe record magnitude?
            if self.record_data:
                self.n_times_randomised += 1
            return control_command_dict

        # DAgger (on control command label).
        d_thrust = self.network_command[0] - self.control_command[0]
        d_br_x = self.network_command[1] - self.control_command[1]
        d_br_y = self.network_command[2] - self.control_command[2]
        d_br_z = self.network_command[3] - self.control_command[3]
        if (self.mode == "testing" and self.use_network) \
                or (self.config.execute_nw_predictions
                    and abs(d_thrust) < self.config.fallback_threshold_rates \
                    and abs(d_br_x) < self.config.fallback_threshold_rates \
                    and abs(d_br_y) < self.config.fallback_threshold_rates \
                    and abs(d_br_z) < self.config.fallback_threshold_rates):
            if self.record_data:
                self.n_times_net += 1
            control_command_dict["network"] = self.network_command
            control_command_dict["use_network"] = True
            return control_command_dict

        # for now just return the expert control command to see if everything works as intended/expected
        # (i.e. it should look similar to running sim.py or stuff in run_tests.py)
        if self.record_data:
            self.n_times_expert += 1
        return control_command_dict

    def _prepare_net_inputs(self):
        if not self.network_initialised:
            # return fake input for init
            if self.config.use_imu:
                if self.config.use_pos:
                    n_init_states = 36
                else:
                    n_init_states = 30
                if self.config.imu_no_rot:
                    n_init_states -= 9
                if self.config.imu_no_vels:
                    n_init_states -= 6
                if self.config.no_ref:
                    n_init_states -= 18 if self.config.use_pos else 15
            else:
                if self.config.use_pos:
                    n_init_states = 18
                else:
                    n_init_states = 15
                if self.config.no_ref:
                    n_init_states = 0
            inputs = {"fts": np.zeros((1, self.config.seq_len, self.config.min_number_fts, 5), dtype=np.float32),
                      "state": np.zeros((1, self.config.seq_len, n_init_states), dtype=np.float32)}
            if self.config.attention_fts_type != "none":
                inputs["attention_fts"] = np.zeros((1, self.config.seq_len, self.attention_fts_size), dtype=np.float32)
            if self.config.attention_branching:
                inputs["attention_label"] = np.zeros((1,), dtype=np.float32)
            elif self.config.gate_direction_branching:
                inputs["gate_direction_label"] = np.zeros((1,), dtype=np.float32)
            if self.config.use_images:
                inputs["image"] = np.zeros((1, self.config.seq_len, 300, 400, 3), dtype=np.float32)
            return inputs

        # reference is always used, state estimate if specified in config
        state_inputs = [] if self.config.no_ref else (self.reference_rot + self.reference[7:].tolist())
        if self.config.use_pos:
            state_inputs += self.reference[:3].tolist()
        if self.config.use_imu:
            estimate = ([] if self.config.imu_no_rot else self.state_estimate_rot) + \
                       ([] if self.config.imu_no_vels else self.state_estimate[7:].tolist())
            if self.config.use_pos:
                estimate += self.state_estimate[:3].tolist()
            state_inputs = estimate + state_inputs
        self.state_queue.append(state_inputs)

        # format the state and feature track inputs as numpy arrays for the network
        state_inputs = np.stack(self.state_queue, axis=0)
        feature_inputs = np.stack(
            [np.stack([v for v in self.fts_queue[j].values()]) for j in range(self.config.seq_len)])
        inputs = {"fts": np.expand_dims(feature_inputs, axis=0).astype(np.float32),
                  "state": np.expand_dims(state_inputs, axis=0).astype(np.float32)}
        if self.config.attention_fts_type != "none":
            attention_fts = self.attention_fts_queue
            if self.config.attention_record_all_features:
                attention_fts = [att_f[self.config.attention_fts_type] for att_f in self.attention_fts_queue]
            attention_fts_inputs = np.stack(attention_fts)
            inputs["attention_fts"] = np.expand_dims(attention_fts_inputs, axis=0).astype(np.float32)
        if self.config.attention_branching:
            inputs["attention_label"] = np.array([self.attention_label], dtype=np.float32)
        elif self.config.gate_direction_branching:
            inputs["gate_direction_label"] = np.array([self.gate_direction_label], dtype=np.float32)
        if self.config.use_images:
            inputs["image"] = np.expand_dims(np.stack(self.image_queue, axis=0), axis=0)
        return inputs

    def compute_trajectory_error(self):
        gt_ref = self.reference[:3]
        gt_pos = self.state[:3]
        results = {"gt_ref": gt_ref, "gt_pos": gt_pos}
        return results

    def write_csv_header(self):
        row = [
            "Rollout_idx",
            "Odometry_stamp",
            # GT Position
            "gt_Position_x",
            "gt_Position_y",
            "gt_Position_z",
            "gt_Position_z_error",
            "gt_Orientation_w",
            "gt_Orientation_x",
            "gt_Orientation_y",
            "gt_Orientation_z",
            "gt_V_linear_x",
            "gt_V_linear_y",
            "gt_V_linear_z",
            "gt_V_angular_x",
            "gt_V_angular_y",
            "gt_V_angular_z",
            # VIO Estimate
            "Position_x",
            "Position_y",
            "Position_z",
            "Position_z_error",
            "Orientation_w",
            "Orientation_x",
            "Orientation_y",
            "Orientation_z",
            "V_linear_x",
            "V_linear_y",
            "V_linear_z",
            "V_angular_x",
            "V_angular_y",
            "V_angular_z",
            # Reference state
            "Reference_position_x",
            "Reference_position_y",
            "Reference_position_z",
            "Reference_orientation_w",
            "Reference_orientation_x",
            "Reference_orientation_y",
            "Reference_orientation_z",
            "Reference_v_linear_x",
            "Reference_v_linear_y",
            "Reference_v_linear_z",
            "Reference_v_angular_x",
            "Reference_v_angular_y",
            "Reference_v_angular_z",
            # MPC output with GT Position
            "Gt_control_command_collective_thrust",
            "Gt_control_command_bodyrates_x",
            "Gt_control_command_bodyrates_y",
            "Gt_control_command_bodyrates_z",
            # Net output
            "Net_control_command_collective_thrust",
            "Net_control_command_bodyrates_x",
            "Net_control_command_bodyrates_y",
            "Net_control_command_bodyrates_z",
            "Maneuver_type",
            # High-level attention decision variable for branching
            "Attention_label",
            "Gate_direction_label",
            # Whether we are collidiging at the moment
            "Collision",
        ]

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.mode == "iterative":
            if self.record_test_data:
                root_save_dir = self.config.val_dir
            else:
                root_save_dir = self.config.train_dir
        else:
            root_save_dir = self.config.log_dir

        self.csv_filename = os.path.join(root_save_dir, "data_" + current_time + ".csv")
        self.image_save_dir = os.path.join(root_save_dir, "img_data_" + current_time)

        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)
        with open(self.csv_filename, "w") as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows([row])

        if self.config.attention_record_all_features:
            self.attention_fts_save_dir = {
                att_f_t: os.path.join(root_save_dir, "{}_{}".format(att_f_t, current_time))
                for att_f_t in ["encoder_fts", "map_tracks", "gaze_tracks"]
            }
            for _, fts_dir in self.attention_fts_save_dir.items():
                if not os.path.exists(fts_dir):
                    os.makedirs(fts_dir)
        elif self.config.attention_fts_type != "none":
            self.attention_fts_save_dir = os.path.join(root_save_dir, "{}_{}".format(
                self.config.attention_fts_type, current_time))
            if not os.path.exists(self.attention_fts_save_dir):
                os.makedirs(self.attention_fts_save_dir)

    def save_data(self):
        row = [
            self.rollout_idx,
            self.simulation_time,  # time stamp
            # GT position
            self.state[0],  # pos x
            self.state[1],  # pos y
            self.state[2],  # pos z
            self.reference[2] - self.state[2],  # ref z - pos z
            self.state[3],  # rot w
            self.state[4],  # rot x
            self.state[5],  # rot y
            self.state[6],  # rot z
            self.state[7],  # vel x
            self.state[8],  # vel y
            self.state[9],  # vel z
            self.state[10],  # omega x
            self.state[11],  # omega y
            self.state[12],  # omega z
            # VIO Estimate
            self.state_estimate[0],  # pos x
            self.state_estimate[1],  # pos y
            self.state_estimate[2],  # pos z
            self.reference[2] - self.state_estimate[2],  # ref z - pos z
            self.state_estimate[3],  # rot w
            self.state_estimate[4],  # rot x
            self.state_estimate[5],  # rot y
            self.state_estimate[6],  # rot z
            self.state_estimate[7],  # vel x
            self.state_estimate[8],  # vel y
            self.state_estimate[9],  # vel z
            self.state_estimate[10],  # omega x
            self.state_estimate[11],  # omega y
            self.state_estimate[12],  # omega z
            # Reference state
            self.reference[0],  # pos x
            self.reference[1],  # pos y
            self.reference[2],  # pos z
            self.reference[3],  # rot w
            self.reference[4],  # rot x
            self.reference[5],  # rot y
            self.reference[6],  # rot z
            self.reference[7],  # vel x
            self.reference[8],  # vel y
            self.reference[9],  # vel z
            self.reference[10],  # omega x
            self.reference[11],  # omega y
            self.reference[12],  # omega z
            # MPC output with GT Position
            self.control_command[0],  # collective thrust
            self.control_command[1],  # roll
            self.control_command[2],  # pitch
            self.control_command[3],  # yaw
            # NET output with GT Position
            self.network_command[0],  # collective thrust
            self.network_command[1],  # roll
            self.network_command[2],  # pitch
            self.network_command[3],  # yaw
            # Maneuver type
            0,
            # High-level attention decision variable for branching
            self.attention_label,
            self.gate_direction_label,
            # Whether we are colliding at the moment
            self.collision,
            ]

        if self.record_data:
            # save the state data and commands
            with open(self.csv_filename, "a") as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows([row])

            # save the feature track data
            # if self.config.use_fts_tracks: TODO: should rework data loading for tensorflow so that this isn't needed
            fts_name = "{:08d}.npy"
            fts_filename = os.path.join(self.image_save_dir, fts_name.format(self.recorded_samples))
            np.save(fts_filename, self.feature_tracks)

            # save attention feature data if specified
            if self.config.attention_record_all_features:
                for att_f_t, fts_dir in self.attention_fts_save_dir.items():
                    current_fts = [att_f[att_f_t] for att_f in self.attention_fts_queue]
                    attention_fts_filename = os.path.join(fts_dir, "{:08d}.npy".format(self.recorded_samples))
                    np.save(attention_fts_filename, np.stack(current_fts, axis=0))
            elif self.config.attention_fts_type != "none":
                attention_fts_filename = os.path.join(self.attention_fts_save_dir,
                                                      "{:08d}.npy".format(self.recorded_samples))
                np.save(attention_fts_filename, np.stack(self.attention_fts_queue, axis=0))

            if self.config.use_images:
                image_file_name = os.path.join(self.image_save_dir, "{:08d}.jpg".format(self.recorded_samples))
                cv2.imwrite(image_file_name, self.image)

            self.recorded_samples += 1
