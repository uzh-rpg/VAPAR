import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.widgets import Button


def load_data(directory):
    inpath_drone = os.path.join(directory, "drone.csv")
    inpath_laps = os.path.join(directory, "laptimes.csv")

    df_drone = pd.read_csv(inpath_drone)
    df_laps = pd.read_csv(inpath_laps)

    _pos_x = []
    _pos_y = []
    _laps = []
    for index, row in df_laps.iterrows():
        if row["is_valid"] == 1:
            temp = df_drone[df_drone["ts"].between(row["ts_start"], row["ts_end"])]
            _pos_x.append(temp["PositionX"].values)
            _pos_y.append(temp["PositionY"].values)
            _laps.append(row["lap"])

    return _pos_x, _pos_y, _laps


def plot_all_trajectories(args, image_path):
    # first plot all valid trajectories to use for comparison
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    laps = []
    for subject in tqdm(sorted(os.listdir(args.data_root))):
        subject_dir = os.path.join(args.data_root, subject)
        if os.path.isdir(subject_dir) and subject.startswith("s0"):
            pos_x = []
            pos_y = []

            for run in sorted(os.listdir(subject_dir)):
                run_dir = os.path.join(subject_dir, run)
                if os.path.isdir(run_dir) and args.track_name in run:
                    x, y, _ = load_data(run_dir)
                    pos_x.extend(x)
                    pos_y.extend(y)

            laps.append((subject, len(pos_x)))

            if len(pos_x) > 0:
                pos_x = np.concatenate(pos_x)
                pos_y = np.concatenate(pos_y)
                plt.plot(pos_x, pos_y, color="#c2c2c2", alpha=0.5)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("tight")
    ax.axis("off")
    ax.set_xlim(-37.14723358154297, 37.202500915527345)
    ax.set_ylim(-19.70142822265625, 21.74341659545899)
    fig.savefig(image_path, dpi=300, frameon="false")

    print("Saved figure of all (valid) trajectories to '{}'.".format(image_path))


class Index:

    def __init__(self, data_root, track_name):
        self.data_root = data_root
        self.track_name = track_name

        self._subject_index = 0
        self._subject_list = list(os.listdir(data_root))
        self._subject_list = sorted([os.path.join(self.data_root, s) for s in self._subject_list])
        self._subject_list = [s for s in self._subject_list if os.path.isdir(s)]

        self._run_index = 0
        self._run_list = None

        self._lap_index = 0
        self._x_pos_list = None
        self._y_pos_list = None
        self._lap_list = None

        self._plot = None
        self._text = None

        self._results = []

    def _run_setup(self):
        # print("_run_setup")
        if self._run_index >= len(self._run_list):
            return

        r_dir = self._run_list[self._run_index]
        # print("_run_setup success, loading from {}".format(r_dir))
        self._x_pos_list, self._y_pos_list, self._lap_list = load_data(r_dir)
        self._lap_index = 0

    def _subject_setup(self):
        # print("_subject_setup")
        if self._subject_index >= len(self._subject_list):
            return

        # print("_subject_setup success")
        s_dir = self._subject_list[self._subject_index]
        self._run_list = list(os.listdir(s_dir))
        self._run_list = sorted([os.path.join(s_dir, r) for r in self._run_list
                                 if os.path.isdir(os.path.join(s_dir, r)) and self.track_name in r])
        self._run_list = [r for r in self._run_list if os.path.isdir(r)]
        self._run_index = 0

    def _find_non_empty_run(self):
        # print("Trying to find non-empty run:")
        # print("len(self._x_pos_list):", len(self._x_pos_list))
        # print("self._run_index:", self._run_index)
        # print("self._lap_index:", self._lap_index)

        while self._x_pos_list is None or len(self._x_pos_list) == 0:
            # print("Trying to find non-empty run, currently at _run_index = {}".format(self._run_index))
            self._run_index += 1

            if self._run_index < len(self._run_list):
                self._run_setup()

            # check if we are done with a subject, then move on to the next
            if self._run_index >= len(self._run_list):
                self._subject_index += 1
                self._subject_setup()

            # check if we are done with all subjects, then close everything
            if self._subject_index >= len(self._subject_list):
                plt.close("all")

    def _click(self, keep):
        if self._run_list is None:
            # prepare run list and load data for first subject and run
            self._subject_setup()
            self._run_setup()
            self._find_non_empty_run()

            # plot the first lap
            self._plot, = plt.plot(self._x_pos_list[self._lap_index], self._y_pos_list[self._lap_index], c="blue")
            self._text = plt.text(-36, 20, "{} - lap {}".format(self._run_list[self._run_index], self._lap_index))
        else:
            # record the result
            self._results.append((self._lap_list[self._lap_index], 1 if keep else 0))
            self._lap_index += 1

            # if it's the end of a run, save the data to the run directory and reset _run_index
            if self._lap_index >= len(self._x_pos_list):
                # print("Test 1")
                results_path = os.path.join(self._run_list[self._run_index], "expected_trajectory.csv")
                df_results = pd.DataFrame(self._results, columns=["lap", "expected_trajectory"])
                df_results.to_csv(results_path, index=False)
                print(df_results)
                print("Saved results to {}.\n".format(results_path))

                self._results = []
                self._lap_index = 0
                self._run_index += 1
                self._run_setup()
                self._find_non_empty_run()

            # check if we are done with a subject, then move on to the next
            if self._run_index >= len(self._run_list):
                # print("Test 2")
                self._run_index = 0
                self._subject_index += 1
                self._subject_setup()
                self._run_setup()
                self._find_non_empty_run()

            # check if we are done with all subjects, then close everything
            if self._subject_index >= len(self._subject_list):
                # print("Test 3")
                plt.close("all")
                return

            # print("New data:")
            # print("len(self._x_pos_list):", len(self._x_pos_list))
            # print("self._run_index:", self._run_index)
            # print("self._lap_index:", self._lap_index)

            # update the plot with the next trajectory
            self._plot.set_xdata(self._x_pos_list[self._lap_index])
            self._plot.set_ydata(self._y_pos_list[self._lap_index])
            self._text.set_text("{} - lap {}".format(self._run_list[self._run_index], self._lap_index))
            plt.draw()

    def click_yes(self, event):
        self._click(True)

    def click_no(self, event):
        self._click(False)


def main(args, image_path):
    # proceed with recording the "expected trajectory" data
    callback = Index(ARGS.data_root, ARGS.track_name)
    image = plt.imread(IMAGE_PATH)

    figure, axis = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    axis.imshow(image, extent=[-37.14723358154297, 37.202500915527345, -19.70142822265625, 21.74341659545899])

    ax_yes = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_no = plt.axes([0.81, 0.05, 0.1, 0.075])

    btn_yes = Button(ax_yes, "Yes")
    btn_yes.on_clicked(callback.click_yes)

    btn_no = Button(ax_no, "No")
    btn_no.on_clicked(callback.click_no)

    plt.sca(axis)
    plt.show()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    PARSER.add_argument("-tn", "--track_name", type=str, default="flat",
                        help="The name of the track.")

    ARGS = PARSER.parse_args()

    # create the overlaid image if it doesn't exist already
    IMAGE_LOAD_DIR = os.path.join(ARGS.data_root, "preprocessing_info")
    IMAGE_PATH = os.path.join(IMAGE_LOAD_DIR, f"all_trajectories_{ARGS.track_name}.png")
    if not os.path.exists(IMAGE_PATH):
        if not os.path.exists(IMAGE_LOAD_DIR):
            os.makedirs(IMAGE_LOAD_DIR)

        print("Need to create image of all overlaid (valid) trajectories first.")
        plot_all_trajectories(ARGS, IMAGE_PATH)

    main(ARGS, IMAGE_PATH)
