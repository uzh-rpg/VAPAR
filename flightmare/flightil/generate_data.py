import os
import numpy as np
import pandas as pd
import cv2
import time

from typing import Type
from tqdm import tqdm

from gazesim.data.utils import iterate_directories, pair

from envs.racing_env_wrapper import RacingEnvWrapper
from planning.planner import TrajectoryPlanner
from planning.mpc_solver import MPCSolver
from features.feature_tracker import FeatureTracker


def row_to_state(row):

    state_original = np.array([
        row["position_x [m]"],
        row["position_y [m]"],
        row["position_z [m]"],  # + 0.75,
        row["rotation_w [quaternion]"],
        row["rotation_x [quaternion]"],
        row["rotation_y [quaternion]"],
        row["rotation_z [quaternion]"],
        row["velocity_x [m/s]"],
        row["velocity_y [m/s]"],
        row["velocity_z [m/s]"],
    ], dtype=np.float32)

    return state_original


def sample_from_trajectory(trajectory, time_stamp):
    # probably just take the closest ts for now, might do interpolation later
    if time_stamp < trajectory["time-since-start [s]"].min():
        index = 0
    else:
        index = trajectory.loc[trajectory["time-since-start [s]"] <= time_stamp, "time-since-start [s]"].idxmax()
    return row_to_state(trajectory.iloc[index])
    # return np.array([0, 0, 5, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    # return row_to_state(trajectory.iloc[0])


def ensure_quaternion_consistency(trajectory, use_norm=True, show_progress=False):
    trajectory = trajectory.reset_index(drop=True)
    flipped = 0
    trajectory["flipped"] = 0
    trajectory.loc[0, "flipped"] = flipped

    quat_columns = ["rotation_{} [quaternion]".format(c) for c in ["w", "x", "y", "z"]]
    prev_quaternion = trajectory.loc[0, quat_columns]
    prev_signs_positive = prev_quaternion >= 0

    norm_diffs = []
    for i in tqdm(range(1, len(trajectory.index)), disable=(not show_progress)):
        current_quaternion = trajectory.loc[i, quat_columns]
        current_signs_positive = current_quaternion >= 0
        condition_sign = prev_signs_positive == ~current_signs_positive

        # TODO: should probably do something like X standard deviations above the running mean for "good methodology"
        norm_diff = np.linalg.norm(prev_quaternion.values - current_quaternion.values)
        norm_diffs.append(norm_diff)

        if use_norm:
            if norm_diff >= 0.5:  # TODO should this be 1.0?
                flipped = 1 - flipped
        else:
            if np.sum(condition_sign) >= 3:
                flipped = 1 - flipped
        trajectory.loc[i, "flipped"] = flipped

        prev_signs_positive = current_signs_positive
        prev_quaternion = current_quaternion

    trajectory.loc[trajectory["flipped"] == 1, quat_columns] *= -1.0

    return trajectory  # , norm_diffs


class DataGenerator:

    def __init__(self, config):
        self.run_dir_list = iterate_directories(config["data_root"], track_names=config["track_name"])
        if config["directory_index"] is not None:
            self.run_dir_list = self.run_dir_list[int(config["directory_index"][0]):config["directory_index"][1]]

        # for r_idx, r in enumerate(self.run_dir_list):
        #     print(r_idx, ":", r)
        # exit()

    def compute_new_data(self, run_dir):
        raise NotImplementedError()

    def finish(self):
        pass

    def generate(self):
        for rd in tqdm(self.run_dir_list, disable=True):
            self.compute_new_data(rd)
        self.finish()


class FlightmareReplicator(DataGenerator):

    COLUMN_DICT = {
        "ts": "time-since-start [s]",
        "PositionX": "position_x [m]",
        "PositionY": "position_y [m]",
        "PositionZ": "position_z [m]",
        "VelocityX": "velocity_x [m/s]",
        "VelocityY": "velocity_y [m/s]",
        "VelocityZ": "velocity_z [m/s]",
        # "AccX": "acceleration_x [m/s/s]",
        # "AccY": "acceleration_y [m/s/s]",
        # "AccZ": "acceleration_z [m/s/s]",
        "rot_w_quat": "rotation_w [quaternion]",
        "rot_x_quat": "rotation_x [quaternion]",
        "rot_y_quat": "rotation_y [quaternion]",
        "rot_z_quat": "rotation_z [quaternion]",
        # "RotationX": "euler_x [rad]",
        # "RotationY": "euler_y [rad]",
        # "RotationZ": "euler_z [rad]",
        # "AngularX": "omega_x [rad/s]",
        # "AngularY": "omega_y [rad/s]",
        # "AngularZ": "omega_z [rad/s]",
        # "DroneVelocityX": "drone_velocity_x [m/s]",
        # "DroneVelocityY": "drone_velocity_y [m/s]",
        # "DroneVelocityZ": "drone_velocity_z [m/s]",
        # "DroneAccelerationX": "drone_acceleration_x [m/s/s]",
        # "DroneAccelerationY": "drone_acceleration_y [m/s/s]",
        # "DroneAccelerationZ": "drone_acceleration_z [m/s/s]",
        # "DroneAngularX": "drone_omega_x [rad]",
        # "DroneAngularY": "drone_omega_y [rad]",
        # "DroneAngularZ": "drone_omega_z [rad]",
    }

    def __init__(self, config):
        super().__init__(config)
        self.skip_existing = config["skip_existing"]
        self.trajectory_only = config["trajectory_only"]
        self.fps = config["frames_per_second"]

        assert 60 % self.fps == 0, "Original FPS (60) should be divisible by new FPS ({}).".format(self.fps)

        self.frame_skip = int(60 / self.fps)

        self.wave_track = config["track_name"] == "wave"
        # self.env = MPCTestWrapper(wave_track=self.wave_track)
        self.env = RacingEnvWrapper()
        if not self.trajectory_only:
            self.env.connect_unity(pub_port=config["pub_port"], sub_port=config["sub_port"])

    def finish(self):
        if not self.trajectory_only:
            self.env.disconnect_unity()

    def compute_new_data(self, run_dir):
        start = time.time()

        # check that the thingy is working
        video_capture = cv2.VideoCapture(os.path.join(run_dir, "screen.mp4"))
        w, h = video_capture.get(3), video_capture.get(4)
        if not (w == 800 and h == 600):
            print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
            return

        print("Processing '{}'.".format(run_dir))

        # load the correct drone.csv and laptimes.csv
        inpath_drone = os.path.join(run_dir, "drone.csv")
        inpath_ts = os.path.join(run_dir, "screen_timestamps.csv")

        df_ts = pd.read_csv(inpath_ts)
        df_traj = pd.read_csv(inpath_drone)

        # select columns and use new column headers
        df_traj = df_traj[[co for co in FlightmareReplicator.COLUMN_DICT]]
        df_traj = df_traj.rename(FlightmareReplicator.COLUMN_DICT, axis=1)

        # ensure quaternion consistency
        df_traj = ensure_quaternion_consistency(df_traj, show_progress=True)

        # adjust height to be suitable for Flightmare
        # df_traj["position_z [m]"] += 0.35  # TODO: THIS IS MAYBE A BUG

        # save the adjusted trajectory (including setting start to 0? probably not)
        # df_traj["time-since-start [s]"] = df_traj["time-since-start [s]"] - df_traj["time-since-start [s]"].min()
        df_traj.to_csv(os.path.join(run_dir, "trajectory.csv"), index=False)

        if self.trajectory_only:
            print("Processed '{}'. in {:.2f}s".format(run_dir, time.time() - start))
            return

        # use this (not time-adjusted) trajectory to generate data
        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, "flightmare_{}.mp4".format(self.fps)),
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            float(self.fps),
            (800, 600),
            True
        )

        # if there are screen timestamps < the first drone timestamp, should just "wait" in the first position
        # for i in tqdm(range(0, len(df_ts.index), self.frame_skip), disable=False):
        for i in tqdm(range(0, 2000, self.frame_skip), disable=False):
            time_current = df_ts["ts"].iloc[i]
            sample = sample_from_trajectory(df_traj, time_current)
            # image = self.env.step(sample)
            self.env.set_reduced_state(sample)
            self.env.render()
            image = self.env.get_image()
            video_writer.write(image)

        """
        for _, row in tqdm(df_ts.iterrows(), total=len(df_ts.index)):
            time_current = row["ts"]
            sample = sample_from_trajectory(df_traj, time_current)
            image = self.env.step(sample)
            video_writer.write(image)
        """

        video_writer.release()

        print("Processed '{}'. in {:.2f}s".format(run_dir, time.time() - start))

        time.sleep(0.1)


class MPCReplicator_v1(DataGenerator):

    # TODO: use the new simulation interface for DDA once its properly done (I think?)
    #  => only "problem" with that might be that images might e.g. not be returned at the correct rate
    #  => could e.g. implement a mode where images/states are queued until a new command has to be returned

    def __init__(self, config):
        super().__init__(config)
        self.skip_existing = config["skip_existing"]
        self.trajectory_only = config["trajectory_only"]
        self.disconnect = config["unity_disconnect"]

        self.fps = config["frames_per_second"]
        self.frame_skip = int(60 / self.fps)
        assert 60 % self.fps == 0, "Original FPS (60) should be divisible by new FPS ({}).".format(self.fps)

        self.command_frequency = config["command_frequency"]
        self.command_skip = int(60 / self.command_frequency)
        assert 60 % self.command_frequency == 0, "For convenience original FPS (60) should be divisible " \
                                                 "by command frequency ({}).".format(self.command_frequency)

        self.base_frequency = 60

        # TODO: probably use integer stuff here
        self.base_time_step = 1.0 / self.base_frequency
        self.image_time_step = 1.0 / self.fps
        self.command_time_step = 1.0 / self.command_frequency

        self.plan_time_step = 0.1
        self.plan_time_horizon = 3.0

        self.mpc_solver = MPCSolver(self.plan_time_horizon, self.plan_time_step)
        self.wave_track = config["track_name"] == "wave"
        self.fm_wrapper = RacingEnvWrapper(wave_track=self.wave_track)
        self.pub_port = config["pub_port"]
        self.sub_port = config["sub_port"]
        if not self.disconnect:
            self.fm_wrapper.connect_unity(self.pub_port, self.sub_port)

    def finish(self):
        self.fm_wrapper.disconnect_unity()

    def compute_new_data(self, run_dir):
        # from run_tests import ensure_quaternion_consistency, visualise_actions, visualise_states

        start = time.time()

        # check that the thingy is working
        video_capture = cv2.VideoCapture(os.path.join(run_dir, "screen.mp4"))
        w, h = video_capture.get(3), video_capture.get(4)
        if not (w == 800 and h == 600):
            print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
            return

        print("\nProcessing '{}'.".format(run_dir))

        # load the correct trajectory and the timestamps which we want to "match"
        inpath_ts = os.path.join(run_dir, "screen_timestamps.csv")
        inpath_traj = os.path.join(run_dir, "trajectory.csv")

        df_ts = pd.read_csv(inpath_ts)
        df_traj = pd.read_csv(inpath_traj)
        df_traj["time-since-start [s]"] = df_traj["time-since-start [s]"]

        # planner
        print("Ensuring quaternion consistency in planner...")
        planner = TrajectoryPlanner(inpath_traj, self.plan_time_horizon, self.plan_time_step,
                                    correct_height_flightmare=False)

        # environment
        self.fm_wrapper.set_reduced_state(planner.get_initial_state())
        self.fm_wrapper.set_sim_time_step(self.base_time_step)
        if self.disconnect:
            self.fm_wrapper.connect_unity(pub_port=self.pub_port, sub_port=self.sub_port)

        # use this (not time-adjusted) trajectory to generate data
        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, "flightmare_mpc_new_{}_{}.mp4".format(self.fps, self.command_frequency)),
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            float(self.fps),
            (self.fm_wrapper.image_width, self.fm_wrapper.image_height),
            True
        )

        # "reset" everything
        reduced_state = planner.get_initial_state()
        state = np.array(reduced_state.tolist() + ([0.0] * 15))
        action = None

        # data to record
        time_stamps = []
        frames = []
        states = []
        actions = []

        print("Running main loop...")
        for _, row in tqdm(df_ts.iterrows(), total=len(df_ts.index), disable=False):
            time_stamps.append(row["ts"])
            frames.append(int(row["frame"]))
            states.append(state)

            if row["frame"] % self.frame_skip == 0:
                image = self.fm_wrapper.get_image()
                video_writer.write(image)

            if row["frame"] % self.command_skip == 0:
                planned_trajectory = np.array(planner.plan(reduced_state, row["ts"]))
                action, predicted_trajectory, cost = self.mpc_solver.solve(planned_trajectory)
            self.fm_wrapper.step(action)
            state = self.fm_wrapper.get_state().copy()
            reduced_state = state[:10]

            actions.append(action)

        video_writer.release()

        states = np.vstack(states)
        actions = np.vstack(actions)

        data = {
            "time-since-start [s]": time_stamps,
            "frame": frames,
            "throttle": actions[:, 0],
            "roll": actions[:, 1],
            "pitch": actions[:, 2],
            "yaw": actions[:, 3],
            "position_x [m]": states[:, 0],
            "position_y [m]": states[:, 1],
            "position_z [m]": states[:, 2],
            "rotation_w [quaternion]": states[:, 3],
            "rotation_x [quaternion]": states[:, 4],
            "rotation_y [quaternion]": states[:, 5],
            "rotation_z [quaternion]": states[:, 6],
            "velocity_x [m/s]": states[:, 7],
            "velocity_y [m/s]": states[:, 8],
            "velocity_z [m/s]": states[:, 9],
            "omega_x [rad/s]": states[:, 10],
            "omega_y [rad/s]": states[:, 11],
            "omega_z [rad/s]": states[:, 12],
            "acceleration_x [m/s/s]": states[:, 13],
            "acceleration_y [m/s/s]": states[:, 14],
            "acceleration_z [m/s/s]": states[:, 15],
        }
        data = pd.DataFrame(data)
        data.to_csv(os.path.join(run_dir, "trajectory_mpc_new_{}.csv".format(self.command_frequency)), index=False)

        print("Processed '{}'. in {:.2f}s".format(run_dir, time.time() - start))

        if self.disconnect:
            self.fm_wrapper.disconnect_unity()


class FeatureTrackGenerator(DataGenerator):

    def __init__(self, config):
        super().__init__(config)
        self.video_name = config["video_name"]
        self.max_features = config["max_features"]
        self.skip_existing = config["skip_existing"]

    def compute_new_data(self, run_dir):
        start = time.time()

        # check that the video exists
        if not os.path.exists(os.path.join(run_dir, "{}.mp4".format(self.video_name))):
            print("WARNING: Video '{}' does not exist for directory '{}'.".format(self.video_name, run_dir))
            return

        video_capture = cv2.VideoCapture(os.path.join(run_dir, "{}.mp4".format(self.video_name)))
        w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))
        assert 60 % int(fps) == 0, "Original FPS (60) should be divisible by new FPS ({}).".format(int(fps))
        frame_skip = int(60 / fps)

        if self.video_name == "screen":
            if not (w == 800 and h == 600):
                print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
                return

        print("Processing '{}'.".format(run_dir))

        # create tracker
        tracker = FeatureTracker(max_features_to_track=self.max_features)

        # get the timestamps (for velocity) and frame index
        # => could probably also enable frame_skip/lower fps this way
        inpath_ts = os.path.join(run_dir, "screen_timestamps.csv")
        df_ts = pd.read_csv(inpath_ts)

        # TODO: need some way of
        #  - saving features for every frame (probably incremental numpy array with one dim being max_features)
        #  - indexing this stuff properly => should we loop through screen_timestamps.csv instead of just video frames?

        # loop through the frames
        features = []
        for i in tqdm(range(0, len(df_ts.index), frame_skip), disable=False):
            frame = df_ts["frame"].iloc[i]
            time_current = df_ts["ts"].iloc[i]

            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame = video_capture.read()
            if not ret:
                print("Could not read frame {} for video '{}' in directory '{}'.".format(i, self.video_name, run_dir))
                continue

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features_current = tracker.process_image(frame_gray, time_current)  # shape (#features, 6)

            if features_current is None:
                features_current = np.empty((0, 6))

            features.append(features_current)

        np.savez(os.path.join(run_dir, "ft_{}.npz".format(self.video_name)), *features)

        # making sure that data exists should be covered by rgb_available,
        # which is used to filter in generate_splits by default

        print("Processed '{}'. in {:.2f}s".format(run_dir, time.time() - start))


def resolve_generator_class(new_data_type: str) -> Type[DataGenerator]:
    # TODO: reference GT
    if new_data_type == "copy_original":
        return FlightmareReplicator
    elif new_data_type == "mpc":
        return MPCReplicator_v1
    elif new_data_type == "mpc_old":
        return MPCReplicator_v0
    elif new_data_type == "feature_tracks":
        return FeatureTrackGenerator
    return DataGenerator


def main(args):
    config = vars(args)

    if config["print_directories_only"]:
        for r_idx, r in enumerate(iterate_directories(config["data_root"], track_names=config["track_name"])):
            print(r_idx, ":", r)
    else:
        generator = resolve_generator_class(config["new_data_type"])(config)
        generator.generate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-tn", "--track_name", type=str, default="flat", choices=["flat", "wave"],
                        help="The track name (relevant for which simulation is used).")
    # TODO: do something about this for stuff that doesn't require a simulation
    parser.add_argument("-ndt", "--new_data_type", type=str, default="copy_original",
                        choices=["copy_original", "mpc", "mpc_old", "feature_tracks"],
                        help="The method to use to compute the ground-truth.")

    parser.add_argument("-vn", "--video_name", type=str, default="screen",
                        help="The name of the videos to use e.g. for computing feature track data.")
    parser.add_argument("-fps", "--frames_per_second", type=int, default=60,
                        help="FPS to use when replicating the data.")
    parser.add_argument("-cf", "--command_frequency", type=int, default=20,
                        help="Frequency at which to compute new control inputs for the MPC.")
    parser.add_argument("-mf", "--max_features", type=int, default=200,
                        help="Maximum number of features to track with the feature tracker.")
    # TODO: maybe option to mask off corner for screen.mp4? but then again, we won't really use those videos anymore

    parser.add_argument("-pp", "--pub_port", type=int, default=10253)
    parser.add_argument("-sp", "--sub_port", type=int, default=10254)
    parser.add_argument("-udc", "--unity_disconnect", action="store_true")

    parser.add_argument("-di", "--directory_index", type=pair, default=None)
    parser.add_argument("-se", "--skip_existing", action="store_true")  # TODO?
    parser.add_argument("-to", "--trajectory_only", action="store_true")
    parser.add_argument("-mo", "--mpc_only", action="store_true")  # TODO: implement and test how quickly it works
    parser.add_argument("-pdo", "--print_directories_only", action="store_true")

    # parse the arguments
    arguments = parser.parse_args()

    # generate the GT
    main(arguments)

