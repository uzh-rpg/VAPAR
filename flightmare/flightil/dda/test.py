import os
import numpy as np
import pandas as pd
import cv2
import argparse
import time
import shutil

from dda.simulation import FlightmareSimulation
from dda.learner import ControllerLearning
from dda.config.settings import create_settings

from gazesim.models.utils import image_softmax


def save_trajectory_data(time_stamps, output_file, mpc_actions, network_actions,
                         states, network_used, save_path, extra_info=None):
    data = {
        "time-since-start [s]": time_stamps,
        "throttle_mpc": mpc_actions[:, 0],
        "roll_mpc": mpc_actions[:, 1],
        "pitch_mpc": mpc_actions[:, 2],
        "yaw_mpc": mpc_actions[:, 3],
        "throttle_nw": network_actions[:, 0],
        "roll_nw": network_actions[:, 1],
        "pitch_nw": network_actions[:, 2],
        "yaw_nw": network_actions[:, 3],
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
        "network_used": np.array(network_used).astype(int)
    }
    if extra_info is not None:
        data.update(extra_info)
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(save_path, "{}.csv".format(output_file)), index=False)


def find_paths(model_load_path, trajectory_path):
    # it is assumed that the file structure is the same as that which DDA creates

    # expand vars in the paths, so that the same file can be used on multiple machines with the same file structure
    new_mlp = []
    for mlp in model_load_path:
        new_mlp.append(os.path.expandvars(mlp))
    model_load_path = new_mlp

    # root directory (to save everything in as well)
    root_dirs = []
    for mlp in model_load_path:
        # take care of getting the right files outside of this if dirs are specified
        root_dirs.append(os.path.abspath(os.path.join(mlp, os.pardir, os.pardir)))
    # root_dir = os.path.abspath(os.path.join(model_load_path, os.pardir, os.pardir))

    # get the actual files to load from
    model_load_path_no_ext = []
    for mlp in model_load_path:
        model_load_path_no_ext.append(os.path.splitext(mlp)[0])
    # model_load_path_no_ext = os.path.splitext(model_load_path)[0]

    # settings file
    settings_files = []
    for rd in root_dirs:
        for file in os.listdir(rd):
            if file.endswith(".yaml"):
                settings_files.append(os.path.join(rd, file))
                break
    # settings_file = None
    # for file in os.listdir(root_dir):
    #     if file.endswith(".yaml"):
    #         settings_file = os.path.join(root_dir, file)
    #         break

    # save dir for the test trajectories
    save_dirs = []
    for rd, sf in zip(root_dirs, settings_files):
        model_name = os.path.basename(sf)
        model_name = model_name.split(".")[0].replace("snaga_", "")
        save_dirs.append(os.path.join(rd, f"dda_{model_name}"))
    # model_name = os.path.basename(settings_file)
    # model_name = model_name.split(".")[0].replace("snaga_", "")
    # save_dir = os.path.join(root_dir, f"dda_{model_name}")

    # figure out whether it is a single trajectory or multiple
    trajectory_paths = []
    if os.path.isfile(trajectory_path) and trajectory_path.endswith(".csv"):
        trajectory_paths.append(os.path.abspath(trajectory_path))
    elif os.path.isdir(trajectory_path):
        for file in os.listdir(trajectory_path):
            if file.startswith("trajectory") and file.endswith(".csv"):
                trajectory_paths.append(os.path.abspath(os.path.join(trajectory_path, file)))
    else:
        raise FileNotFoundError("Path '{}' is not a valid trajectory file or folder".format(trajectory_path))

    return root_dirs, model_load_path_no_ext, settings_files, save_dirs, trajectory_paths


def main(args):
    root_dirs, model_load_paths_no_ext, settings_files, save_dirs, trajectory_paths = find_paths(
        args.model_load_path, args.trajectory_path)

    """
    from pprint import pprint
    pprint(root_dirs)
    print()
    pprint(model_load_paths_no_ext)
    print()
    pprint(settings_files)
    print()
    pprint(save_dirs)

    root_dirs = root_dirs[:2]
    model_load_paths_no_ext = model_load_paths_no_ext[:2]
    settings_files = settings_files[:2]
    save_dirs = save_dirs[:2]
    """

    # want to keep the same simulation instance if this is to be used on snaga
    # (where disconnecting from the Unity application leads to an error), therefore
    # use it for all models/trajectories (the timeout should be set accordingly)
    simulation = None

    print(settings_files)

    # TODO: also record
    #  - gaze prediction + 3D vector
    #  - drone velocity is already recorded I guess
    #  - maybe current reference though? for plotting the reference trajectory?
    #  - current high-level-label

    # loop over all models
    experiments_total = len(root_dirs) * len(trajectory_paths) * args.repetitions
    experiments_counter = 0
    for root_dir, model_load_path_no_ext, settings_file, save_dir in zip(
            root_dirs, model_load_paths_no_ext, settings_files, save_dirs):
        model_start = time.time()
        print("\n[Testing] Starting testing for '{}'\n".format(root_dir))

        # create the directory to save the outputs in if it doesn't exist already
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # copy the settings file to the save directory
        shutil.copy(settings_file, save_dir)

        # create and modify settings
        settings = create_settings(settings_file, mode="dagger", generate_log=False)
        settings.resume_training = True
        settings.resume_ckpt_file = model_load_path_no_ext
        settings.gpu = args.gpu
        settings.flightmare_pub_port = args.pub_port
        settings.flightmare_sub_port = args.sub_port
        settings.max_time = 1000.0
        if args.offline_evaluation:
            settings.start_buffer = 1000.0
        if args.record_extra_info:
            settings.return_extra_info = True

        # using "learner" as controller
        controller = ControllerLearning(settings, trajectory_paths[0], mode="testing", max_time=settings.max_time)

        if simulation is None:
            # create simulation (do it here so we don't disconnect and mess things up on snaga)
            simulation = FlightmareSimulation(settings, trajectory_paths[0], max_time=settings.max_time)

            # connect to the simulation either at the start or after training has been run
            simulation.connect_unity(settings.flightmare_pub_port, settings.flightmare_sub_port)

            # wait until Unity rendering/image queue has calmed down
            for _ in range(50):
                simulation.flightmare_wrapper.get_image()
                time.sleep(0.1)
        else:
            # hopefully this works as intended, in principle nothing should change for all models we are testing for
            # but if this is changed later, that might not be the case (similar to the track type being set)
            simulation.update_config(settings)

        # test for each specified trajectory:
        for trajectory_path in trajectory_paths:
            trajectory_start = time.time()
            print("\n[Testing] Starting testing for '{}'\n".format(trajectory_path))

            # determine the directory to save the output in
            trajectory_name = os.path.basename(trajectory_path)
            trajectory_name = trajectory_name.split(".")[0]
            trajectory_dir = os.path.join(save_dir, trajectory_name)
            if not os.path.exists(trajectory_dir):
                os.makedirs(trajectory_dir)

            # copy the original trajectory file to that folder for reference
            shutil.copyfile(trajectory_path, os.path.join(trajectory_dir, "original.csv"))

            # update the simulation and learner, which contain trajectory samplers/planners
            simulation.update_trajectory(trajectory_path, max_time=settings.max_time)
            controller.update_trajectory(trajectory_path, max_time=settings.max_time)

            # repeatedly fly the current trajectory
            for repetition in range(args.repetitions):
                experiments_counter += 1
                repetition_start = time.time()
                print("\n[Testing] Starting repetition {} ({}/{})\n".format(
                    repetition, experiments_counter, experiments_total))

                # file name(s)
                if args.output_file is not None:
                    output_file = "{}_{:02d}".format(args.output_file, repetition)
                else:
                    if args.offline_evaluation:
                        output_file = "mpc_nw_act_{:02d}".format(repetition)
                    else:
                        output_file = "mpc2nw_mt-{:02d}_st-{:02d}_{:02d}".format(
                            int(simulation.total_time * 10), int(settings.start_buffer * 10), repetition)

                writer = None
                if args.save_video:
                    writer = cv2.VideoWriter(
                        os.path.join(trajectory_dir, "{}.mp4".format(output_file)),
                        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                        settings.base_frequency,
                        (simulation.flightmare_wrapper.image_width, simulation.flightmare_wrapper.image_height),
                        True,
                    )

                fts_writer = None
                if args.save_feature_track_video:
                    fts_writer = cv2.VideoWriter(
                        os.path.join(trajectory_dir, "fts_{}.mp4".format(output_file)),
                        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                        settings.image_frequency,
                        (simulation.flightmare_wrapper.image_width, simulation.flightmare_wrapper.image_height),
                        True,
                    )

                att_writer = None
                if args.save_attention_video:
                    att_writer = cv2.VideoWriter(
                        os.path.join(trajectory_dir, "att_{}.mp4".format(output_file)),
                        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                        settings.base_frequency,
                        (simulation.flightmare_wrapper.image_width, simulation.flightmare_wrapper.image_height),
                        True,
                    )

                all_features = {}
                colors = np.random.randint(0, 255, (controller.feature_tracker.max_features_to_track, 3))

                # data to record
                time_stamps = []
                states = []
                mpc_actions = []
                network_actions = []
                network_used = []
                extra_info = None
                if args.record_extra_info:
                    extra_info = []

                # whether to use the network instead of the MPC
                use_network = False

                # resetting everything
                trajectory_done = False
                info_dict = simulation.reset()
                controller.reset()
                controller.use_network = use_network
                controller.record_data = False
                controller.update_info(info_dict)
                controller.prepare_network_command()
                controller.prepare_expert_command()
                action = controller.get_control_command()

                # run the main loop until the simulation "signals" that the trajectory is done
                while not trajectory_done:
                    # decide whether to switch to network at the current time
                    if info_dict["time"] > settings.start_buffer:
                        use_network = True
                        controller.use_network = use_network

                    # print(info_dict["reference"][:3])

                    # record states
                    time_stamps.append(info_dict["time"])
                    states.append(info_dict["state"])

                    # record actions
                    mpc_actions.append(action["expert"] if not use_network or args.record_mpc_actions
                                       else np.array([np.nan] * 4))
                    network_actions.append(action["network"])
                    network_used.append(action["use_network"])

                    if info_dict["update"]["image"] and args.save_feature_track_video:
                        current_image = info_dict["image"].copy()
                        mask = np.zeros_like(current_image)
                        current_features = controller.feature_tracks

                        # for f_idx, f in enumerate(current_features):
                        for f_id, feat in current_features.items():
                            point = tuple(((feat[0:2] + 1) / 2 * np.array([800.0, 600.0])).astype(int))
                            if f_id not in all_features:
                                all_features[f_id] = [point]
                            else:
                                all_features[f_id].append(point)

                        # TODO: only iterate over the stuff that's in the current dict
                        # for f_idx, f in enumerate(current_features):
                        color_idx = 0
                        for f_id, feat in current_features.items():
                            points = all_features[f_id]
                            for i in range(len(points) - 1):
                                mask = cv2.line(mask, (points[i][0], points[i][1]),
                                                (points[i + 1][0], points[i + 1][1]),
                                                colors[color_idx].tolist(), 2)
                            current_image = cv2.circle(
                                current_image, (points[-1][0], points[-1][1]), 5, colors[color_idx].tolist(), -1)
                            color_idx += 1

                        # pprint(controller.feature_tracks)
                        # print()
                        current_image = cv2.add(current_image, mask)
                        # cv2.imshow("frame", current_image)
                        # cv2.waitKey(0)
                        fts_writer.write(current_image)

                    if args.record_extra_info:
                        extra_info.append(controller.extra_info.copy())

                    if args.save_video:
                        writer.write(info_dict["image"])

                    if args.save_attention_video:
                        att = controller.extra_info["out_attention"]
                        att = image_softmax(att)
                        att = att.cpu().detach().numpy().squeeze()
                        if att.max() != 0:
                            att /= att.max()
                        att = (att * 255).astype("uint8")
                        att = np.repeat(att[np.newaxis, :, :], 3, axis=0).transpose((1, 2, 0))
                        att[:, :, :-1] = 0
                        att = cv2.resize(att, (800, 600))
                        att = cv2.addWeighted(info_dict["image"], 1.0, att, 1.0, 0)
                        att_writer.write(att)

                    # perform the step(s) in the simulation and get the new action
                    info_dict = simulation.step(action["network"] if action["use_network"] else action["expert"])
                    if use_network and not args.record_mpc_actions:
                        info_dict["update"]["expert"] = False

                    trajectory_done = info_dict["done"]
                    if not trajectory_done:
                        controller.update_info(info_dict)
                        if not settings.save_at_net_frequency or info_dict["update"]["command"]:
                            action = controller.get_control_command()

                # prepare data
                states = np.vstack(states)
                mpc_actions = np.vstack(mpc_actions)
                network_actions = np.vstack(network_actions)

                if args.record_extra_info:
                    extra_info = {k: [d[k] for d in extra_info] for k in extra_info[0]}

                if args.save_video:
                    writer.release()

                if args.save_feature_track_video:
                    fts_writer.release()

                if args.save_attention_video:
                    att_writer.release()

                # save the data
                # trajectory_dir = "/home/simon/Desktop/weekly_meeting/meeting21/debug_weird_nw_pred"
                save_trajectory_data(time_stamps, output_file, mpc_actions, network_actions,
                                     states, network_used, trajectory_dir, extra_info)

                print("\n[Testing] Finished repetition {} in {:.2f}s ({}/{})\n".format(
                    repetition, time.time() - repetition_start, experiments_counter, experiments_total))

            print("\n[Testing] Finished testing for '{}' in {:.2f}s\n".format(
                trajectory_path, time.time() - trajectory_start))

        print("\n[Testing] Finished testing for '{}' in {:.2f}s\n".format(root_dir, time.time() - model_start))

    if simulation is not None:
        simulation.disconnect_unity()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Network", fromfile_prefix_chars="@")
    parser.add_argument("-mlp", "--model_load_path", type=str, nargs="+", required=True,
                        help="Path(s) to model checkpoint. Can be listed in a plain text file and "
                             "read from the file by specifying '@filename' for this argument.")
    parser.add_argument("-tp", "--trajectory_path", type=str, required=True, help="Path to trajectory/trajectories")
    parser.add_argument("-of", "--output_file", type=str, help="Output file name other than default trajectory")
    parser.add_argument("-rep", "--repetitions", type=int, default=20, help="Repetitions for testing")
    parser.add_argument("-pp", "--pub_port", type=int, default=10253, help="Flightmare publisher port")
    parser.add_argument("-sp", "--sub_port", type=int, default=10254, help="Flightmare subscriber port")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU to run networks on")
    parser.add_argument("-rma", "--record_mpc_actions", action="store_true", help="Whether or not to do this")
    parser.add_argument("-off", "--offline_evaluation", action="store_true",
                        help="Whether or not to evaluate the trained model 'offline', i.e. using the MPC to "
                             "fly the trajectory, but evaluating the model is done during training and "
                             "recording its actions.")
    parser.add_argument("-xi", "--record_extra_info", action="store_true",
                        help="Whether or not to record extra info, e.g. from attention branching (only one "
                             "implemented right now) or feature tracks or IMU.")
    parser.add_argument("-sv", "--save_video", action="store_true",
                        help="Whether or not to save the frames as a video.")
    parser.add_argument("-sftv", "--save_feature_track_video", action="store_true",
                        help="Whether or not to save the frames with feature tracks as a video.")
    parser.add_argument("-sav", "--save_attention_video", action="store_true",
                        help="Whether or not to save the frames attention predictions overlaid as a video. "
                             "Note that this option was only used temporarily to create some visualisations and "
                             "for it to work some of the rest of the code was changed as well. The latter changes "
                             "have now been reversed, thus this will not actually work.")

    main(parser.parse_args())
