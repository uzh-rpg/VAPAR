import os
import numpy as np
import pandas as pd
import cv2
import torch

from typing import Type
from tqdm import tqdm
from time import time
from gazesim.data.utils import iterate_directories, generate_gaussian_heatmap, filter_by_screen_ts, parse_run_info, pair
from gazesim.training.helpers import resolve_dataset_class
from gazesim.training.utils import load_model, to_device, to_batch
from gazesim.models.utils import image_softmax, image_log_softmax


class DataGenerator:

    def __init__(self, config):
        self.run_dir_list = iterate_directories(config["data_root"], track_names=config["track_name"])
        if config["directory_index"] is not None:
            self.run_dir_list = self.run_dir_list[int(config["directory_index"][0]):config["directory_index"][1]]
        """
        for r_idx, r in enumerate(self.run_dir_list):
            print(r_idx, ":", r)
        exit()
        """

    def get_gt_info(self, run_dir, subject, run):
        pass

    def compute_gt(self, run_dir):
        raise NotImplementedError()

    def generate(self):
        for rd in tqdm(self.run_dir_list, disable=True):
            self.compute_gt(rd)


class MovingWindowFrameMeanGT(DataGenerator):

    NAME = "moving_window_frame_mean_gt"

    def __init__(self, config):
        super().__init__(config)

        # make sure the input is correct
        assert config["mw_size"] > 0 and config["mw_size"] % 2 == 1

        self._half_window_size = int((config["mw_size"] - 1) / 2)
        self._skip_existing = config["skip_existing"]

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        gaze_gt_path = os.path.join(index_dir, "gaze_gt.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(gaze_gt_path):
            df_gaze_gt = pd.read_csv(gaze_gt_path)
        else:
            df_gaze_gt = df_frame_index.copy()
            df_gaze_gt = df_gaze_gt[["frame", "subject", "run"]]

        if self.__class__.NAME not in df_gaze_gt.columns:
            df_gaze_gt[self.__class__.NAME] = -1

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_gaze_gt["subject"] == subject) & (df_gaze_gt["run"] == run)

        return df_gaze_gt, gaze_gt_path, match_index

    def compute_gt(self, run_dir):
        start = time()

        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_gaze_gt, gaze_gt_path, match_index = self.get_gt_info(run_dir, subject, run)

        # initiate video capture and writer
        video_file = os.path.join(run_dir, "screen.mp4")
        if os.path.isfile(video_file):
            video_capture = cv2.VideoCapture(video_file)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))

            if not (w == 800 and h == 600):
                print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
                return

            if not int(num_frames) == match_index.sum():
                print("WARNING: Number of frames in video and registered in main index is different for directory '{}'.".format(run_dir))
                return
        else:
            print("WARNING: No 'screen.mp4' video found for directory '{}'. Using default values for video parameters.".format(run_dir))
            w = 800
            h = 600
            fps = 60.0
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            num_frames = match_index.sum()

        # load data frames with the timestamps and positions for gaze and the frames/timestamps for the video
        df_gaze = pd.read_csv(os.path.join(run_dir, "gaze_on_surface.csv"))
        df_screen = pd.read_csv(os.path.join(run_dir, "screen_timestamps.csv"))

        # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
        df_gaze = df_gaze[["ts", "frame", "norm_x_su", "norm_y_su"]]
        df_gaze.columns = ["ts", "frame", "x", "y"]
        df_gaze["x"] = df_gaze["x"] * 800
        df_gaze["y"] = (1.0 - df_gaze["y"]) * 600

        # filter by screen timestamps
        df_screen, df_gaze = filter_by_screen_ts(df_screen, df_gaze)

        # since the measurements are close together and to reduce computational load,
        # compute the mean measurement for each frame
        # TODO: think about whether this is a good way to deal with this issue
        df_gaze = df_gaze[["frame", "x", "y"]]
        df_gaze = df_gaze.groupby("frame").mean()
        df_gaze["frame"] = df_gaze.index

        # create new dataframe to write frame-level information into
        df_gaze_gt.loc[match_index, self.__class__.NAME] = df_screen["frame"].isin(df_gaze["frame"]).astype(int).values

        # save gaze gt index to CSV with updated data
        df_gaze_gt.to_csv(gaze_gt_path, index=False)

        if self._skip_existing and os.path.exists(os.path.join(run_dir, f"{self.__class__.NAME}.mp4")):
            print("INFO: Video already exists for '{}'.".format(run_dir))
            return

        # writer is only initialised after making sure that everything else works
        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, f"{self.__class__.NAME}.mp4"),
            int(fourcc),
            fps,
            (int(w), int(h)),
            True
        )

        # loop through all frames and compute the ground truth (where possible)
        for frame_idx in tqdm(df_screen["frame"], disable=False):
            if frame_idx >= num_frames:
                print("Number of frames in CSV file exceeds number of frames in video!")
                break

            # compute the range of frames to use for the ground truth
            frame_low = frame_idx - self._half_window_size
            frame_high = frame_idx + self._half_window_size

            # create the heatmap
            current_frame_data = df_gaze[df_gaze["frame"].between(frame_low, frame_high)]
            current_mu = current_frame_data[["x", "y"]].values
            heatmap = generate_gaussian_heatmap(mu=current_mu, down_scale_factor=10)

            if heatmap.max() > 0.0:
                heatmap /= heatmap.max()

            heatmap = (heatmap * 255).astype("uint8")
            heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)

            # save the resulting frame
            video_writer.write(heatmap)

        video_writer.release()

        # TODO: copy and adapt this to saving position-only GT

        print("Saved moving window ground-truth for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


class FrameMeanGazeGT(DataGenerator):

    NAME = "frame_mean_gaze_gt"

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        gaze_gt_path = os.path.join(index_dir, "gaze_gt.csv")
        gaze_measurements_path = os.path.join(index_dir, f"{self.__class__.NAME}.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(gaze_gt_path):
            df_gaze_gt = pd.read_csv(gaze_gt_path)
        else:
            df_gaze_gt = df_frame_index.copy()
            df_gaze_gt = df_gaze_gt[["frame", "subject", "run"]]

        if os.path.exists(gaze_measurements_path):
            df_gaze_measurements = pd.read_csv(gaze_measurements_path)
        else:
            df_gaze_measurements = df_frame_index.copy()
            df_gaze_measurements = df_gaze_measurements[["frame"]]
            df_gaze_measurements.columns = ["x"]
            df_gaze_measurements["x"] = np.nan
            df_gaze_measurements["y"] = np.nan

        if self.__class__.NAME not in df_gaze_gt.columns:
            df_gaze_gt[self.__class__.NAME] = -1

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_gaze_gt["subject"] == subject) & (df_gaze_gt["run"] == run)

        return df_gaze_gt, df_gaze_measurements, gaze_gt_path, gaze_measurements_path, match_index

    def compute_gt(self, run_dir):
        start = time()

        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_gaze_gt, df_gaze_measurements, gaze_gt_path, gaze_measurements_path, match_index = self.get_gt_info(
            run_dir, subject, run)

        # initiate video capture and writer
        video_file = os.path.join(run_dir, "screen.mp4")
        if os.path.isfile(video_file):
            video_capture = cv2.VideoCapture(video_file)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))

            if not (w == 800 and h == 600):
                print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
                return
        else:
            print("WARNING: No 'screen.mp4' video found for directory '{}'. Using default values for video parameters.".format(run_dir))

        # load data frames with the timestamps and positions for gaze and the frames/timestamps for the video
        df_gaze = pd.read_csv(os.path.join(run_dir, "gaze_on_surface.csv"))
        df_screen = pd.read_csv(os.path.join(run_dir, "screen_timestamps.csv"))

        # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
        df_gaze = df_gaze[["ts", "frame", "norm_x_su", "norm_y_su"]]
        df_gaze.columns = ["ts", "frame", "x", "y"]
        df_gaze["x"] = df_gaze["x"] * 2.0 - 1.0
        # df_gaze["x"] = df_gaze["x"].clip(-1.0, 1.0)
        df_gaze["y"] = (1.0 - df_gaze["y"]) * 2.0 - 1.0
        # df_gaze["y"] = df_gaze["y"].clip(-1.0, 1.0)
        df_gaze["frame"] = -1

        # TODO: if there are values that are nan between values that aren't then it could just be the case
        #  that for that frame, there aren't any measurements (for the timestamps around that frame)
        #  => this should happen very rarely, but should probably be corrected
        #  => does it make sense after all to use measurements from more than one frame??

        """
        print(df_gaze[["x", "y"]].min())
        print(df_gaze[["x", "y"]].max())
        print(df_gaze[["x", "y"]].mean())
        print(df_gaze[["x", "y"]].median())
        print((df_gaze["x"] < -1.0).sum(), (df_gaze["x"] > 1.0).sum())
        print((df_gaze["y"] < -1.0).sum(), (df_gaze["y"] > 1.0).sum())
        print(len(df_gaze.index))
        print()
        exit()
        """

        # filter by screen timestamps
        df_screen, df_gaze = filter_by_screen_ts(df_screen, df_gaze)

        # since the measurements are close together and to reduce computational load,
        # compute the mean measurement for each frame
        df_gaze = df_gaze[["frame", "x", "y"]]
        df_gaze = df_gaze.groupby("frame").mean()
        df_gaze["frame"] = df_gaze.index
        df_gaze = df_gaze.reset_index(drop=True)

        # add information about gaze GT being available to frame-wise screen info
        df_gaze_gt.loc[match_index, self.__class__.NAME] = df_screen["frame"].isin(df_gaze["frame"]).astype(int).values
        df_gaze_columns = df_gaze_measurements.copy()[["x", "y"]]
        df_gaze = df_gaze.set_index("frame")
        for (_, row), f in tqdm(zip(df_gaze.iterrows(), df_gaze.index), disable=False, total=len(df_gaze.index)):
            df_gaze_columns.iloc[match_index & (df_gaze_gt["frame"] == f)] = row.values
        df_gaze_measurements[["x", "y"]] = df_gaze_columns[["x", "y"]]

        # save gaze gt to CSV with updated data
        df_gaze_gt.to_csv(gaze_gt_path, index=False)
        df_gaze_measurements.to_csv(gaze_measurements_path, index=False)

        print("Saved frame mean gaze GT for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


class ShuffledRandomFrameMeanGazeGT(DataGenerator):

    NAME = "shuffled_random_frame_mean_gaze_gt"

    def __init__(self, config):
        super().__init__(config)

        # load dataframes
        self.frame_index = pd.read_csv(os.path.join(config["data_root"], "index", "frame_index.csv"))
        self.fmg_gt_or = pd.read_csv(os.path.join(config["data_root"], "index", "frame_mean_gaze_gt.csv"))
        self.fmg_gt = self.fmg_gt_or.copy()
        self.fmg_gt_path = os.path.join(config["data_root"], "index", f"{self.__class__.NAME}.csv")

        np.random.seed(config["random_seed"])

    def compute_gt(self, run_dir):
        start = time()

        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]
        match_index = (self.frame_index["subject"] == subject) & (self.frame_index["run"] == run)

        if subject < 6:
            print("WARNING: Data from faulty subject will not be processed for directory '{}'.".format(run_dir))
            return

        # load data frames
        df_screen = pd.read_csv(os.path.join(run_dir, "screen_timestamps.csv"))
        df_laps = pd.read_csv(os.path.join(run_dir, "laptimes.csv"))

        # determine intervals
        intervals = []
        for _, row in df_laps.iterrows():
            if len(intervals) > 0 and abs(intervals[-1][1] - row["ts_start"]) > 1e-6:
                intervals.append((intervals[-1][1], row["ts_start"]))
            intervals.append((row["ts_start"], row["ts_end"]))
        if len(intervals) == 0:
            intervals.append((df_screen["ts"].iloc[0], df_screen["ts"].iloc[-1]))
        else:
            if df_screen["ts"].iloc[0] < intervals[0][0]:
                intervals.insert(0, (df_screen["ts"].iloc[0], intervals[0][0]))
            if df_screen["ts"].iloc[-1] > intervals[-1][1]:
                intervals.append((intervals[-1][1], df_screen["ts"].iloc[-1]))

        # loop through the intervals
        progress_bar = tqdm(total=len(intervals))
        new_values = []
        for itv_idx, (itv_start, itv_end) in enumerate(intervals):
            # get the frames
            if itv_idx != len(intervals) - 1:
                match_upper = df_screen["ts"] < itv_end
            else:
                match_upper = df_screen["ts"] <= itv_end
            frames = df_screen.loc[(itv_start <= df_screen["ts"]) & match_upper, "frame"].values

            # shuffle the frames
            np.random.shuffle(frames)

            # loop through them in that order
            for f in frames:
                # get values at current position and append them to the list in new order
                current_values = self.fmg_gt_or.loc[match_index & (self.frame_index["frame"] == f)].values[0]
                new_values.append(current_values)
                progress_bar.update(1)

        self.fmg_gt.loc[match_index, :] = np.array(new_values)
        self.fmg_gt.to_csv(os.path.join(self.fmg_gt_path), index=False)

        print("Saved shuffled random frame mean gaze ground-truth for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


class RandomGazeGT(DataGenerator):

    NAME = "random_gaze_gt"

    def __init__(self, config):
        super().__init__(config)

        # make sure the input is correct
        assert config["mw_size"] > 0 and config["mw_size"] % 2 == 1

        self._half_window_size = int((config["mw_size"] - 1) / 2)
        self._skip_existing = config["skip_existing"]

        np.random.seed(config["random_seed"])

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        gaze_gt_path = os.path.join(index_dir, "test_gaze_gt.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(gaze_gt_path):
            df_gaze_gt = pd.read_csv(gaze_gt_path)
        else:
            df_gaze_gt = df_frame_index.copy()
            df_gaze_gt = df_gaze_gt[["frame", "subject", "run"]]

        if self.__class__.NAME not in df_gaze_gt.columns:
            df_gaze_gt[self.__class__.NAME] = -1

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_gaze_gt["subject"] == subject) & (df_gaze_gt["run"] == run)

        return df_gaze_gt, gaze_gt_path, match_index

    def compute_gt(self, run_dir):
        start = time()

        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_gaze_gt, gaze_gt_path, match_index = self.get_gt_info(run_dir, subject, run)

        # initiate video capture and writer
        video_file = os.path.join(run_dir, "screen.mp4")
        if os.path.isfile(video_file):
            video_capture = cv2.VideoCapture(video_file)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))

            if not (w == 800 and h == 600):
                print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
                return

            if not int(num_frames) == match_index.sum():
                print("WARNING: Number of frames in video and registered in main index "
                      "is different for directory '{}'.".format(run_dir))
                return
        else:
            print("WARNING: No 'screen.mp4' video found for directory '{}'. Using default values for video parameters.".format(run_dir))
            w = 800
            h = 600
            fps = 60.0
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            num_frames = match_index.sum()

        # moved this here to be able to exit as early as possible if video already exists
        # this "ground-truth" type is available for all frames
        df_gaze_gt.loc[match_index, self.__class__.NAME] = 1

        # save gaze gt index to CSV with updated data
        df_gaze_gt.to_csv(gaze_gt_path, index=False)

        if self._skip_existing and os.path.exists(os.path.join(run_dir, f"{self.__class__.NAME}.mp4")):
            print("INFO: Video already exists for '{}'.".format(run_dir))
            return

        # load data frames with the timestamps and positions for gaze and the frames/timestamps for the video
        df_screen = pd.read_csv(os.path.join(run_dir, "screen_timestamps.csv"))
        df_screen["x"] = w / 2
        df_screen["y"] = h / 2

        # initial position
        next_pos = np.random.multivariate_normal(np.array([h / 2, w / 2]), np.array([[h, 0.0], [0.0, w]]))
        while not (0.0 <= next_pos[0] < h and 0.0 <= next_pos[1] < w):
            next_pos = np.random.multivariate_normal(np.array([h / 2, w / 2]), np.array([[h, 0.0], [0.0, w]]))
        df_screen.loc[df_screen.index[0], "x"] = next_pos[1]
        df_screen.loc[df_screen.index[0], "y"] = next_pos[0]
        prev_pos = next_pos

        # iteratively adding positions for every frame
        for i in tqdm(range(1, len(df_screen.index)), disable=False):
            next_pos = np.random.multivariate_normal(prev_pos, np.array([[h / 8, 0.0], [0.0, w / 8]]))
            while not (0.0 <= next_pos[0] < h and 0.0 <= next_pos[1] < w):
                next_pos = np.random.multivariate_normal(prev_pos, np.array([[h / 8, 0.0], [0.0, w / 8]]))

            df_screen.loc[df_screen.index[i], "x"] = next_pos[1]
            df_screen.loc[df_screen.index[i], "y"] = next_pos[0]

            prev_pos = next_pos

        df_screen["x"] = df_screen["x"].astype(int)
        df_screen["y"] = df_screen["y"].astype(int)

        # writer is only initialised after making sure that everything else works
        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, f"{self.__class__.NAME}.mp4"),
            int(fourcc),
            fps,
            (int(w), int(h)),
            True
        )

        # loop through all frames and "create" the ground truth
        for frame_idx in tqdm(df_screen["frame"], disable=False):
            if frame_idx >= num_frames:
                print("Number of frames in CSV file exceeds number of frames in video!")
                break

            # compute the range of frames to use for the ground truth
            frame_low = frame_idx - self._half_window_size
            frame_high = frame_idx + self._half_window_size

            # create the heatmap
            current_frame_data = df_screen[df_screen["frame"].between(frame_low, frame_high)]
            current_mu = current_frame_data[["x", "y"]].values
            heatmap = generate_gaussian_heatmap(mu=current_mu, down_scale_factor=10)

            if heatmap.max() > 0.0:
                heatmap /= heatmap.max()

            heatmap = (heatmap * 255).astype("uint8")
            heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)

            # save the resulting frame
            video_writer.write(heatmap)

        video_writer.release()

        print("Saved random gaze ground-truth for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


class ShuffledRandomGazeGT(DataGenerator):

    NAME = "shuffled_random_{}"

    def __init__(self, config):
        super().__init__(config)

        self.video_name = config["video_name"]
        self.class_name = self.__class__.NAME.format(self.video_name)
        np.random.seed(config["random_seed"])

    def compute_gt(self, run_dir):
        start = time()

        # initiate video capture
        video_file = os.path.join(run_dir, f"{self.video_name}.mp4")
        if os.path.isfile(video_file):
            video_capture = cv2.VideoCapture(video_file)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))

            if not (w == 800 and h == 600):
                print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
                return
        else:
            print(f"WARNING: No '{self.video_name}.mp4' video found for directory '{run_dir}'. "
                  f"Using default values for video parameters.")
            w = 800
            h = 600
            fps = 60.0
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")

        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, f"{self.class_name}.mp4"),
            int(fourcc),
            fps,
            (int(w), int(h)),
            True
        )

        # load data frames
        df_screen = pd.read_csv(os.path.join(run_dir, "screen_timestamps.csv"))
        df_laps = pd.read_csv(os.path.join(run_dir, "laptimes.csv"))

        # determine intervals
        intervals = []
        for _, row in df_laps.iterrows():
            if len(intervals) > 0 and abs(intervals[-1][1] - row["ts_start"]) > 1e-6:
                intervals.append((intervals[-1][1], row["ts_start"]))
            intervals.append((row["ts_start"], row["ts_end"]))
        if len(intervals) == 0:
            intervals.append((df_screen["ts"].iloc[0], df_screen["ts"].iloc[-1]))
        else:
            if df_screen["ts"].iloc[0] < intervals[0][0]:
                intervals.insert(0, (df_screen["ts"].iloc[0], intervals[0][0]))
            if df_screen["ts"].iloc[-1] > intervals[-1][1]:
                intervals.append((intervals[-1][1], df_screen["ts"].iloc[-1]))

        # loop through the intervals
        progress_bar = tqdm(total=len(intervals))
        for itv_idx, (itv_start, itv_end) in enumerate(intervals):
            # get the frames
            if itv_idx != len(intervals) - 1:
                match_upper = df_screen["ts"] < itv_end
            else:
                match_upper = df_screen["ts"] <= itv_end
            frames = df_screen.loc[(itv_start <= df_screen["ts"]) & match_upper, "frame"].values

            # shuffle the frames
            np.random.shuffle(frames)

            # loop through them in that order
            for f in frames:
                # load frame at shuffled location, then save it
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, f)
                _, image = video_capture.read()
                video_writer.write(image)
                progress_bar.update(1)

        video_capture.release()
        video_writer.release()

        print("Saved shuffled random gaze ground-truth for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


class PredictedGazeGT(DataGenerator):

    NAME = "predicted_gaze_gt"

    def __init__(self, config):
        super().__init__(config)

        self.video_name = config["video_name"]
        self.batch_size = config["batch_size"]

        # load the model and config here
        self.model, self.config = load_model(config["model_load_path"], gpu=config["gpu"], return_config=True)
        self.device = torch.device("cuda:{}".format(self.config["gpu"]) if
                                   torch.cuda.is_available() and
                                   self.config["gpu"] < torch.cuda.device_count() else "cpu")
        self.model.to(self.device)

        # modify the config
        self.config["data_root"] = config["data_root"]

        frame_index = pd.read_csv(os.path.join(config["data_root"], "index", "frame_index.csv"))
        self.config["split_config"] = frame_index["track_name"] \
                                      + "_" + frame_index["subject"].astype(str) \
                                      + "_" + frame_index["run"].astype(str)

        # TODO: best thing to do might be to load a model, get the dataset type etc., construct a dataset
        #  with an "artificial" config (could e.g. change in dataset so if config["split_config"] is already
        #  a dataframe, it just uses that), that is constructed from the frame_index (where rgb_available == True)
        #  and then get the indices for each video and load that stuff from the dataset, construct batches (maybe)
        #  and put it through whichever network is being loaded
        # => can we iterate with dataloader? probs not, since we need subindex right?
        #    => could use batch_sampler argument?
        # if there are stacks, should probably just insert black frames
        # => how to check this, difference between len(dataset) and num_frames

    def get_gt_info(self, run_dir, subject, run):
        pass

    def compute_gt(self, run_dir):
        start = time()

        # get info about the current run
        # TODO: probably not needed
        run_info = parse_run_info(run_dir)
        run_split = "{}_{}_{}".format(run_info["track_name"], run_info["subject"], run_info["run"])

        # initiate video capture and writer
        video_file = os.path.join(run_dir, f"{self.video_name}.mp4")
        num_frames = None
        if os.path.isfile(video_file):
            video_capture = cv2.VideoCapture(video_file)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))
            num_frames = int(num_frames)

            if not (w == 800 and h == 600):
                print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
                return
        else:
            print(f"WARNING: No '{self.video_name}.mp4' video found for directory '{run_dir}'. "
                  f"Using default values for video parameters.")
            w = 800
            h = 600
            fps = 60.0
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")

        # create dataset
        data_set = resolve_dataset_class(self.config["dataset_name"])(self.config, run_split)

        # check that dataset length and number of frames in video match
        if os.path.isfile(video_file) and num_frames != len(data_set):
            print("WARNING: Dataset size ({}) and number of frames in video ({}) not the same for directory '{}'"
                  .format(len(data_set), num_frames, run_dir))
            return

        # writer is only initialised after making sure that everything else works
        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, f"{self.__class__.NAME}.mp4"),
            int(fourcc),
            fps,
            (int(w), int(h)),
            True
        )

        # loop through all frames and compute the attention prediction
        # for frame in tqdm(data_set):
        for start_frame in tqdm(range(0, len(data_set), self.batch_size)):
            batch = []
            for frame_idx in range(start_frame, start_frame + self.batch_size):
                if frame_idx >= len(data_set):
                    break
                batch.append(data_set[frame_idx])
            batch = to_device(to_batch(batch), self.device)
            output = self.model(batch)
            attention_batch = image_softmax(image_log_softmax(output["output_attention"])).cpu().detach().numpy().squeeze(1)
            """
            frame = to_device(frame, self.device, make_batch=True)  # TODO: should probably use larger batch size
            output = self.model(frame)
            attention = image_softmax(image_log_softmax(output["output_attention"])).cpu().detach().numpy().squeeze()
            """

            # print(attention.shape)
            # print(attention.max(), attention.min())

            for attention in attention_batch:
                # adjust the attention map to be an image in the same format as moving_window_frame_mean_gt
                attention = np.repeat(attention[np.newaxis, :, :], 3, axis=0).transpose((1, 2, 0))
                norm_max = np.max(attention)
                if norm_max != 0:
                    attention /= norm_max
                attention = (attention * 255).astype("uint8")
                attention = cv2.resize(attention, (800, 600), interpolation=cv2.INTER_CUBIC)

                # write the attention map
                video_writer.write(attention)

            # exit()

        video_writer.release()

        print("Saved predicted attention maps for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


class OpticalFlowFarneback(DataGenerator):

    NAME = "optical_flow_farneback"

    def __init__(self, config):
        super().__init__(config)

    def get_gt_info(self, run_dir, subject, run):
        # probably just ignore this, since anything where RGB is available is pretty much good?
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        gaze_gt_path = os.path.join(index_dir, "gaze_gt.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(gaze_gt_path):
            df_gaze_gt = pd.read_csv(gaze_gt_path)
        else:
            df_gaze_gt = df_frame_index.copy()
            df_gaze_gt = df_gaze_gt[["frame", "subject", "run"]]

        if self.__class__.NAME not in df_gaze_gt.columns:
            df_gaze_gt[self.__class__.NAME] = -1

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_gaze_gt["subject"] == subject) & (df_gaze_gt["run"] == run)

        return df_gaze_gt, gaze_gt_path, match_index

    def compute_gt(self, run_dir):
        start = time()

        # get info about the current run
        # TODO: probably not needed
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # initiate video capture and writer
        video_capture = cv2.VideoCapture(os.path.join(run_dir, "screen.mp4"))
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))
        num_frames = int(num_frames)

        if not (w == 800 and h == 600):
            print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
            return

        # writer is only initialised after making sure that everything else works
        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, f"{self.__class__.NAME}.mp4"),
            int(fourcc),
            fps,
            (int(w), int(h)),
            True
        )

        # loop through all frames and compute the optical flow
        _, previous_frame = video_capture.read()
        video_writer.write(np.zeros_like(previous_frame))
        hsv_representation = np.zeros_like(previous_frame)
        hsv_representation[..., 1] = 255
        previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        # TODO: should there be some way to indicate whether optical flow is available? e.g. in frame_index?
        for frame_idx in tqdm(range(1, num_frames), disable=False):
            _, current_frame = video_capture.read()
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            optical_flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude, angle = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
            hsv_representation[..., 0] = angle * 180 / np.pi / 2
            hsv_representation[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            rgb_representation = cv2.cvtColor(hsv_representation, cv2.COLOR_HSV2BGR)

            # save the resulting frame
            video_writer.write(rgb_representation)

            previous_frame = current_frame

        video_writer.release()

        print("Saved optical flow for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


class DroneControlFrameMeanGT(DataGenerator):

    NAME = "drone_control_frame_mean_gt"

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        control_gt_path = os.path.join(index_dir, "control_gt.csv")
        control_measurements_path = os.path.join(index_dir, f"{self.__class__.NAME}.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(control_gt_path):
            df_control_gt = pd.read_csv(control_gt_path)
        else:
            df_control_gt = df_frame_index.copy()
            df_control_gt = df_control_gt[["frame", "subject", "run"]]

        oc = ["throttle_norm [0,1]", "roll_norm [-1,1]", "pitch_norm [-1,1]", "yaw_norm [-1,1]"]
        c = ["throttle", "roll", "pitch", "yaw"]
        if os.path.exists(control_measurements_path):
            df_control_measurements = pd.read_csv(control_measurements_path)
            df_control_measurements = df_control_measurements.rename(
                {c: rc for c, rc in zip(oc, c)}, axis=1)
        else:
            df_control_measurements = df_frame_index.copy()
            df_control_measurements = df_control_measurements[["frame"]]
            df_control_measurements.columns = [c[0]]
            df_control_measurements[c[0]] = np.nan
            for col in c:
                if col not in df_control_measurements.columns:
                    df_control_measurements[col] = np.nan

        if self.__class__.NAME not in df_control_gt.columns:
            df_control_gt[self.__class__.NAME] = -1
            for col in c:
                if col not in df_control_measurements.columns:
                    df_control_measurements[col] = np.nan

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_control_gt["subject"] == subject) & (df_control_gt["run"] == run)

        return df_control_gt, df_control_measurements, control_gt_path, control_measurements_path, match_index, c, oc

    def compute_gt(self, run_dir):
        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_control_gt, df_control_measurements, control_gt_path, control_measurements_path, match_index, c, oc = \
            self.get_gt_info(run_dir, subject, run)

        # define paths
        df_drone_path = os.path.join(run_dir, "drone.csv")
        df_screen_path = os.path.join(run_dir, "screen_timestamps.csv")

        # load dataframes
        df_drone = pd.read_csv(df_drone_path)
        df_screen = pd.read_csv(df_screen_path)

        # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
        df_drone = df_drone.rename({c: rc for c, rc in zip(oc, c)}, axis=1)
        df_drone = df_drone[(["ts"] + c)]
        df_drone["frame"] = -1

        # filter by screen timestamps
        df_screen, df_drone = filter_by_screen_ts(df_screen, df_drone)

        # compute the mean measurement for each frame
        df_drone = df_drone[(["frame"] + c)]
        df_drone = df_drone.groupby("frame").mean()
        df_drone["frame"] = df_drone.index
        df_drone = df_drone.reset_index(drop=True)
        df_drone = df_drone[(["frame"] + c)]

        # add information about control GT being available to frame-wise screen info
        df_control_gt.loc[match_index, self.__class__.NAME] = df_screen["frame"].isin(df_drone["frame"]).astype(int).values
        df_control_measurements_columns = df_control_measurements.copy()[c]
        df_drone = df_drone.set_index("frame")
        for (_, row), f in tqdm(zip(df_drone.iterrows(), df_drone.index), disable=False, total=len(df_drone.index)):
            df_control_measurements_columns.iloc[match_index & (df_control_gt["frame"] == f)] = row.values
        df_control_measurements[c] = df_control_measurements_columns[c]

        # save control gt to CSV with updated data
        df_control_gt.to_csv(control_gt_path, index=False)
        df_control_measurements.to_csv(control_measurements_path, index=False)


class DroneControlMPCGT(DataGenerator):

    NAME = "drone_control_mpc_{}_gt"

    def __init__(self, config):
        super(DroneControlMPCGT, self).__init__(config)

        self.command_frequency = config["mpc_command_frequency"]
        self.class_name = self.__class__.NAME.format(self.command_frequency)

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        control_gt_path = os.path.join(index_dir, "control_gt.csv")
        control_measurements_path = os.path.join(index_dir, f"{self.class_name}.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(control_gt_path):
            df_control_gt = pd.read_csv(control_gt_path)
        else:
            df_control_gt = df_frame_index.copy()
            df_control_gt = df_control_gt[["frame", "subject", "run"]]

        oc = ["throttle_norm [0,1]", "roll_norm [-1,1]", "pitch_norm [-1,1]", "yaw_norm [-1,1]"]
        c = ["throttle", "roll", "pitch", "yaw"]
        if os.path.exists(control_measurements_path):
            df_control_measurements = pd.read_csv(control_measurements_path)
            df_control_measurements = df_control_measurements.rename(
                {c: rc for c, rc in zip(oc, c)}, axis=1)  # not sure why this is here...
        else:
            df_control_measurements = df_frame_index.copy()
            df_control_measurements = df_control_measurements[["frame"]]
            df_control_measurements.columns = [c[0]]
            df_control_measurements[c[0]] = np.nan
            for col in c:
                if col not in df_control_measurements.columns:
                    df_control_measurements[col] = np.nan

        if self.class_name not in df_control_gt.columns:
            df_control_gt[self.class_name] = -1
            for col in c:
                if col not in df_control_measurements.columns:
                    df_control_measurements[col] = np.nan

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_control_gt["subject"] == subject) & (df_control_gt["run"] == run)

        return df_control_gt, df_control_measurements, control_gt_path, control_measurements_path, match_index, c, oc

    def compute_gt(self, run_dir):
        if not os.path.exists(os.path.join(run_dir, "trajectory_mpc_{}.csv".format(self.command_frequency))):
            print("WARNING: No MPC trajectory found for directory '{}'.".format(run_dir))
            return

        start = time()

        print("\nProcessing '{}'.".format(run_dir))

        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_control_gt, df_control_measurements, control_gt_path, control_measurements_path, match_index, c, oc = \
            self.get_gt_info(run_dir, subject, run)

        # define paths
        df_screen_path = os.path.join(run_dir, "screen_timestamps.csv")
        df_traj_path = os.path.join(run_dir, "trajectory_mpc_{}.csv".format(self.command_frequency))
        # TODO: need to rename "thrust" to "throttle" in these files
        # TODO: need to read the trajectory_mpc file, get the actions and stuff

        # load dataframes
        df_screen = pd.read_csv(df_screen_path)
        df_traj = pd.read_csv(df_traj_path)

        # can be immediately selected here, since the column names match
        df_traj = df_traj[c]

        # TODO: all of this needs to be changed once the MPC stuff is done better (with a loop through
        #  screen_timestamps.csv) => when we have passed "total_time", should maybe just put nan?
        #  => should just stop writing at that point, we can take care of this with the min_length thing below
        #     everything after that should not be set to 1 in df_control_gt

        # add information about control GT being available to frame-wise screen info
        min_length = min(len(df_screen.index), len(df_traj.index))

        # print(min_length, len(df_screen.index), len(df_traj.index))
        # print(sum(match_index))
        # print("BEFORE")
        # print(df_control_gt.loc[match_index, self.class_name])
        # print(len(df_control_gt.index))

        df_control_gt.loc[match_index, self.class_name] = df_control_gt.loc[match_index, "frame"].isin(
            df_screen.loc[:min_length, "frame"]).astype(int).values
        df_control_measurements_columns = df_control_measurements.copy()[c]

        # print("AFTER")
        # print(df_control_gt.loc[match_index, self.class_name])

        # print(len(df_control_measurements_columns.index))

        for (_, row), f in tqdm(zip(list(df_traj.iterrows())[:min_length],
                                    df_screen["frame"].iloc[:min_length]), total=min_length):
            df_control_measurements_columns.iloc[match_index & (df_control_gt["frame"] == f)] = row.values

        # print(len(df_control_measurements_columns.index))
        df_control_measurements[c] = df_control_measurements_columns[c]
        # print(len(df_control_measurements.index))

        # print(df_control_gt.loc[match_index, self.class_name])
        # print(df_control_measurements_columns.loc[match_index])
        # print(df_control_measurements.loc[match_index])

        # save control gt to CSV with updated data
        df_control_gt.to_csv(control_gt_path, index=False)
        df_control_measurements.to_csv(control_measurements_path, index=False)

        print("Processed '{}'. in {:.2f}s".format(run_dir, time() - start))


class DroneControlFrameMeanRawGT(DataGenerator):

    NAME = "drone_control_frame_mean_raw_gt"

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        control_gt_path = os.path.join(index_dir, "control_gt.csv")
        control_measurements_path = os.path.join(index_dir, f"{self.__class__.NAME}.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(control_gt_path):
            df_control_gt = pd.read_csv(control_gt_path)
        else:
            df_control_gt = df_frame_index.copy()
            df_control_gt = df_control_gt[["frame", "subject", "run"]]

        oc = ["throttle_thrust [N]", "roll_rate [rad/s]", "pitch_rate [rad/s]", "yaw_rate [rad/s]"]
        c = ["throttle", "roll", "pitch", "yaw"]
        if os.path.exists(control_measurements_path):
            df_control_measurements = pd.read_csv(control_measurements_path)
            df_control_measurements = df_control_measurements.rename(
                {c: rc for c, rc in zip(oc, c)}, axis=1)
        else:
            df_control_measurements = df_frame_index.copy()
            df_control_measurements = df_control_measurements[["frame"]]
            df_control_measurements.columns = [c[0]]
            df_control_measurements[c[0]] = np.nan
            for col in c:
                if col not in df_control_measurements.columns:
                    df_control_measurements[col] = np.nan

        if self.__class__.NAME not in df_control_gt.columns:
            df_control_gt[self.__class__.NAME] = -1
            for col in c:
                if col not in df_control_measurements.columns:
                    df_control_measurements[col] = np.nan

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_control_gt["subject"] == subject) & (df_control_gt["run"] == run)

        return df_control_gt, df_control_measurements, control_gt_path, control_measurements_path, match_index, c, oc

    def compute_gt(self, run_dir):
        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_control_gt, df_control_measurements, control_gt_path, control_measurements_path, match_index, c, oc = \
            self.get_gt_info(run_dir, subject, run)

        # define paths
        df_drone_path = os.path.join(run_dir, "drone.csv")
        df_screen_path = os.path.join(run_dir, "screen_timestamps.csv")

        # load dataframes
        df_drone = pd.read_csv(df_drone_path)
        df_screen = pd.read_csv(df_screen_path)

        # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
        df_drone = df_drone.rename({c: rc for c, rc in zip(oc, c)}, axis=1)
        df_drone = df_drone[(["ts"] + c)]
        df_drone["frame"] = -1

        # filter by screen timestamps
        df_screen, df_drone = filter_by_screen_ts(df_screen, df_drone)

        # compute the mean measurement for each frame
        df_drone = df_drone[(["frame"] + c)]
        df_drone = df_drone.groupby("frame").mean()
        df_drone["frame"] = df_drone.index
        df_drone = df_drone.reset_index(drop=True)
        df_drone = df_drone[(["frame"] + c)]

        # add information about control GT being available to frame-wise screen info
        df_control_gt.loc[match_index, self.__class__.NAME] = df_screen["frame"].isin(df_drone["frame"]).astype(int).values
        df_control_measurements_columns = df_control_measurements.copy()[c]
        df_drone = df_drone.set_index("frame")
        for (_, row), f in tqdm(zip(df_drone.iterrows(), df_drone.index), disable=False, total=len(df_drone.index)):
            df_control_measurements_columns.iloc[match_index & (df_control_gt["frame"] == f)] = row.values
        df_control_measurements[c] = df_control_measurements_columns[c]

        # save control gt to CSV with updated data
        df_control_gt.to_csv(control_gt_path, index=False)
        df_control_measurements.to_csv(control_measurements_path, index=False)


class DroneStateFrameMean(DataGenerator):

    # TODO: maybe rename this script, since this isn't technically used as ground-truth?

    NAME = "drone_state_frame_mean"

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        state_path = os.path.join(index_dir, "state.csv")

        # TODO: should it really be the mean? or just the closest value???
        #  => in that case something like the trajectory sampler would be necessary........
        #  => might want to do this stuff in the new repo...

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(state_path):
            df_state = pd.read_csv(state_path)
        else:
            df_state = df_frame_index.copy()
            df_state = df_state[["frame", "subject", "run"]]

        columns = ["DroneVelocityX", "DroneVelocityY", "DroneVelocityZ", "DroneAccelerationX", "DroneAccelerationY",
                   "DroneAccelerationZ", "DroneAngularX", "DroneAngularY", "DroneAngularZ"]
        if columns[0] not in df_state.columns:
            for col in columns:
                df_state[col] = np.nan

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_state["subject"] == subject) & (df_state["run"] == run)

        return df_state, state_path, match_index, columns

    def compute_gt(self, run_dir):
        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_state, state_path, match_index, columns = self.get_gt_info(run_dir, subject, run)

        # define paths
        df_drone_path = os.path.join(run_dir, "drone.csv")
        df_screen_path = os.path.join(run_dir, "screen_timestamps.csv")

        # load dataframes
        df_drone = pd.read_csv(df_drone_path)
        df_screen = pd.read_csv(df_screen_path)

        # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
        df_drone = df_drone[(["ts"] + columns)]
        df_drone["frame"] = -1

        # filter by screen timestamps
        df_screen, df_drone = filter_by_screen_ts(df_screen, df_drone)

        # compute the mean measurement for each frame
        df_drone = df_drone[(["frame"] + columns)]
        df_drone = df_drone.groupby("frame").mean()
        df_drone["frame"] = df_drone.index
        df_drone = df_drone.reset_index(drop=True)
        df_drone = df_drone[(["frame"] + columns)]

        # add information about control GT being available to frame-wise screen info
        df_state_columns = df_state.copy()[columns]
        df_drone = df_drone.set_index("frame")
        for (_, row), f in tqdm(zip(df_drone.iterrows(), df_drone.index), disable=False, total=len(df_drone.index)):
            df_state_columns.iloc[match_index & (df_state["frame"] == f)] = row.values
        df_state[columns] = df_state_columns[columns]

        # save control gt to CSV with updated data
        df_state.to_csv(state_path, index=False)


class DroneStateOriginal(DataGenerator):

    # TODO: maybe rename this script, since this isn't technically used as ground-truth?

    NAME = "drone_state_original"

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        state_index_path = os.path.join(index_dir, "state.csv")
        state_path = os.path.join(index_dir, f"{self.__class__.NAME}.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(state_index_path):
            df_state_index = pd.read_csv(state_index_path)
        else:
            df_state_index = df_frame_index.copy()
            df_state_index = df_state_index[["frame", "subject", "run"]]

        # TODO: this is a horrible way of doing things, should probably be in constants.py or something
        oc = [
            "position_x [m]", "position_y [m]", "position_z [m]",
            "velocity_x [m/s]", "velocity_y [m/s]", "velocity_z [m/s]",
            "acceleration_x [m/s/s]", "acceleration_y [m/s/s]", "acceleration_z [m/s/s]",
            "rotation_w [quaternion]", "rotation_x [quaternion]", "rotation_y [quaternion]", "rotation_z [quaternion]",
            "euler_x [rad]", "euler_y [rad]", "euler_z [rad]",
            "omega_x [rad/s]", "omega_y [rad/s]", "omega_z [rad/s]",
            "drone_velocity_x [m/s]", "drone_velocity_y [m/s]", "drone_velocity_z [m/s]",
            "drone_acceleration_x [m/s/s]", "drone_acceleration_y [m/s/s]", "drone_acceleration_z [m/s/s]",
            "drone_omega_x [rad]", "drone_omega_y [rad]", "drone_omega_z [rad]",
        ]
        c = [
            "position_x", "position_y", "position_z",
            "velocity_x", "velocity_y", "velocity_z",
            "acceleration_x", "acceleration_y", "acceleration_z",
            "rotation_w", "rotation_x", "rotation_y", "rotation_z",
            "euler_x", "euler_y", "euler_z",
            "omega_x", "omega_y", "omega_z",
            "drone_velocity_x", "drone_velocity_y", "drone_velocity_z",
            "drone_acceleration_x", "drone_acceleration_y", "drone_acceleration_z",
            "drone_omega_x", "drone_omega_y", "drone_omega_z",
        ]
        if os.path.exists(state_path):
            df_state = pd.read_csv(state_path)
            df_state = df_state.rename(
                {c: rc for c, rc in zip(oc, c)}, axis=1)  # not sure why this is here...
        else:
            df_state = df_frame_index.copy()
            df_state = df_state[["frame"]]
            df_state.columns = [c[0]]
            df_state[c[0]] = np.nan
            for col in c:
                if col not in df_state.columns:
                    df_state[col] = np.nan

        if self.__class__.NAME not in df_state_index.columns:
            df_state_index[self.__class__.NAME] = -1
            for col in c:
                if col not in df_state.columns:
                    df_state[col] = np.nan

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_state_index["subject"] == subject) & (df_state_index["run"] == run)

        return df_state_index, df_state, state_index_path, state_path, match_index, c, oc

    def compute_gt(self, run_dir):
        if not os.path.exists(os.path.join(run_dir, "trajectory.csv")):
            print("WARNING: No original trajectory found for directory '{}'.".format(run_dir))
            return

        start = time()

        print("\nProcessing '{}'.".format(run_dir))

        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_state_index, df_state, state_index_path, state_path, match_index, c, oc = self.get_gt_info(run_dir, subject, run)

        # define paths
        df_screen_path = os.path.join(run_dir, "screen_timestamps.csv")
        df_traj_path = os.path.join(run_dir, "trajectory.csv")

        # load dataframes
        df_screen = pd.read_csv(df_screen_path)
        df_traj = pd.read_csv(df_traj_path)

        # can't be immediately selected here, since the column names don't match
        df_traj = df_traj.rename({or_c: new_c for or_c, new_c in zip(oc, c)}, axis=1)
        df_traj = df_traj[["time-since-start [s]"] + c]

        # TODO: this is basically what I need for the MPC stuff, but not the original trajectory
        #  => should just loop through screen_timestamps and get the closest state...

        # add information about control GT being available to frame-wise screen info
        df_state_index.loc[match_index, self.__class__.NAME] = 1
        df_state_columns = df_state.copy()[c]

        # add the actual states
        for _, row in tqdm(list(df_screen.iterrows())):
            # sample from the trajectory
            row_idx = df_traj["time-since-start [s]"] <= row["ts"]
            if all(~row_idx):
                index = 0
            else:
                index = df_traj.loc[row_idx, "time-since-start [s]"].idxmax()
            df_state_columns.iloc[match_index & (df_state_index["frame"] == row["frame"])] = df_traj[c].iloc[index]
        df_state[c] = df_state_columns[c]

        # save to CSV with updated data
        df_state_index.to_csv(state_index_path, index=False)
        df_state.to_csv(state_path, index=False)

        print("Processed '{}'. in {:.2f}s".format(run_dir, time() - start))


class DroneStateMPC(DataGenerator):

    NAME = "drone_state_mpc_{}"

    def __init__(self, config):
        super(DroneStateMPC, self).__init__(config)

        self.command_frequency = config["mpc_command_frequency"]
        self.class_name = self.__class__.NAME.format(self.command_frequency)

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        state_index_path = os.path.join(index_dir, "state.csv")
        state_path = os.path.join(index_dir, f"{self.class_name}.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(state_index_path):
            df_state_index = pd.read_csv(state_index_path)
        else:
            df_state_index = df_frame_index.copy()
            df_state_index = df_state_index[["frame", "subject", "run"]]

        oc = [
            "position_x [m]", "position_y [m]", "position_z [m]",
            "velocity_x [m/s]", "velocity_y [m/s]", "velocity_z [m/s]",
            "acceleration_x [m/s/s]", "acceleration_y [m/s/s]", "acceleration_z [m/s/s]",
            "rotation_w [quaternion]", "rotation_x [quaternion]", "rotation_y [quaternion]", "rotation_z [quaternion]",
            "omega_x [rad/s]", "omega_y [rad/s]", "omega_z [rad/s]",
        ]
        c = [
            "position_x", "position_y", "position_z",
            "velocity_x", "velocity_y", "velocity_z",
            "acceleration_x", "acceleration_y", "acceleration_z",
            "rotation_w", "rotation_x", "rotation_y", "rotation_z",
            "omega_x", "omega_y", "omega_z",
        ]
        if os.path.exists(state_path):
            df_state = pd.read_csv(state_path)
            df_state = df_state.rename(
                {c: rc for c, rc in zip(oc, c)}, axis=1)  # not sure why this is here...
        else:
            df_state = df_frame_index.copy()
            df_state = df_state[["frame"]]
            df_state.columns = [c[0]]
            df_state[c[0]] = np.nan
            for col in c:
                if col not in df_state.columns:
                    df_state[col] = np.nan

        if self.class_name not in df_state_index.columns:
            df_state_index[self.class_name] = -1
            for col in c:
                if col not in df_state.columns:
                    df_state[col] = np.nan

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_state_index["subject"] == subject) & (df_state_index["run"] == run)

        return df_state_index, df_state, state_index_path, state_path, match_index, c, oc

    def compute_gt(self, run_dir):
        if not os.path.exists(os.path.join(run_dir, "trajectory_mpc_{}.csv".format(self.command_frequency))):
            print("WARNING: No MPC trajectory found for directory '{}'.".format(run_dir))
            return

        start = time()

        print("\nProcessing '{}'.".format(run_dir))

        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_state_index, df_state, control_gt_path, control_measurements_path, match_index, c, oc = \
            self.get_gt_info(run_dir, subject, run)

        # define paths
        df_screen_path = os.path.join(run_dir, "screen_timestamps.csv")
        df_traj_path = os.path.join(run_dir, "trajectory_mpc_{}.csv".format(self.command_frequency))

        # load dataframes
        df_screen = pd.read_csv(df_screen_path)
        df_traj = pd.read_csv(df_traj_path)

        # can be immediately selected here, since the column names match
        df_traj = df_traj.rename({or_c: new_c for or_c, new_c in zip(oc, c)}, axis=1)
        # df_traj = df_traj[["time-since-start [s]"] + c]
        df_traj = df_traj[c]  # IS THAT STILL THE CASE? NO, THIS ISN'T ABOUT COMMANDS AFTER ALL

        # add information about control GT being available to frame-wise screen info
        # TODO: this still relies on everything being recorded in 60FPS/matching screen_timestamps.csv,
        #  maybe this should be changed at some point to e.g. have higher frequency state estimates
        min_length = min(len(df_screen.index), len(df_traj.index))

        df_state_index.loc[match_index, self.class_name] = df_state_index.loc[match_index, "frame"].isin(
            df_screen.loc[:min_length, "frame"]).astype(int).values
        df_state_columns = df_state.copy()[c]

        # gather the actual data
        for (_, row), f in tqdm(zip(list(df_traj.iterrows())[:min_length],
                                    df_screen["frame"].iloc[:min_length]), total=min_length):
            df_state_columns.iloc[match_index & (df_state_index["frame"] == f)] = row.values
        df_state[c] = df_state_columns[c]

        # save to CSV with updated data
        df_state_index.to_csv(control_gt_path, index=False)
        df_state.to_csv(control_measurements_path, index=False)

        print("Processed '{}'. in {:.2f}s".format(run_dir, time() - start))


def resolve_gt_class(data_type: str) -> Type[DataGenerator]:
    if data_type == "moving_window_frame_mean_gt":
        return MovingWindowFrameMeanGT
    elif data_type == "frame_mean_gaze_gt":
        return FrameMeanGazeGT
    elif data_type == "predicted_gaze_gt":
        return PredictedGazeGT
    elif data_type == "drone_control_frame_mean_gt":
        return DroneControlFrameMeanGT
    elif data_type == "drone_control_frame_mean_raw_gt":
        return DroneControlFrameMeanRawGT
    elif data_type == "drone_control_mpc_gt":
        return DroneControlMPCGT
    elif data_type == "random_gaze_gt":
        return RandomGazeGT
    elif data_type == "shuffled_random_gaze_gt":
        return ShuffledRandomGazeGT
    elif data_type == "shuffled_random_frame_mean_gaze_gt":
        return ShuffledRandomFrameMeanGazeGT
    elif data_type == "drone_state_frame_mean":
        return DroneStateFrameMean
    elif data_type == "drone_state_original":
        return DroneStateOriginal
    elif data_type == "drone_state_mpc":
        return DroneStateMPC
    elif data_type == "optical_flow":
        return OpticalFlowFarneback
    # TODO: attention prediction generator
    return DataGenerator


def main(args):
    config = vars(args)

    if config["print_directories_only"]:
        for r_idx, r in enumerate(iterate_directories(config["data_root"], track_names=config["track_name"])):
            print(r_idx, ":", r)
    else:
        generator = resolve_gt_class(config["data_type"])(config)
        generator.generate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-tn", "--track_name", type=str, nargs="+", default=["flat", "wave"], choices=["flat", "wave"],
                        help="The method to use to compute the ground-truth.")
    parser.add_argument("-dt", "--data_type", type=str, default="moving_window_frame_mean_gt",
                        choices=["moving_window_frame_mean_gt", "drone_control_frame_mean_gt", "drone_control_mpc_gt",
                                 "drone_control_frame_mean_raw_gt", "random_gaze_gt", "shuffled_random_gaze_gt",
                                 "drone_state_frame_mean", "drone_state_original", "drone_state_mpc", "optical_flow",
                                 "predicted_gaze_gt", "frame_mean_gaze_gt", "shuffled_random_frame_mean_gaze_gt"],
                        help="The method to use to compute the ground-truth.")
    parser.add_argument("-di", "--directory_index", type=pair, default=None)
    parser.add_argument("-rs", "--random_seed", type=int, default=127,
                        help="The random seed.")
    parser.add_argument("-se", "--skip_existing", action="store_true")
    parser.add_argument("-pdo", "--print_directories_only", action="store_true")

    # arguments only used for specific data types
    parser.add_argument("--mw_size", type=int, default=25,
                        help="Size of the temporal window in frames from which the "
                             "ground-truth for the current frame should be computed.")

    parser.add_argument("-mcf", "--mpc_command_frequency", type=int, default=20,
                        help="Frequency at which control inputs were computed for MPC drone control GT generation .")

    parser.add_argument("-vn", "--video_name", type=str, default="screen",
                        help="The name of the videos to use e.g. for computing feature track data.")
    parser.add_argument("-mlp", "--model_load_path", type=str,
                        help=".")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help=".")
    parser.add_argument("-g", "--gpu", type=int, default=0)
    # FPS? if the input video is different, should use that FPS I guess
    # batch size? would probably speed things up significantly

    # parse the arguments
    arguments = parser.parse_args()

    # generate the GT
    main(arguments)

