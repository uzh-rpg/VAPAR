# NOTE: currently, these stem from splits/split004_info.json and are computed over all
#       frames that have drone measurements and drone_control_frame_mean_gt available
STATISTICS = {
    "mean": {
        "screen": [0.23185605229941816, 0.20987627008239895, 0.21252105159994594],
        "hard_mask_moving_window_frame_mean_gt": [0.23185605229941816, 0.20987627008239895, 0.21252105159994594]
    },
    "std": {
        "screen": [0.14304103712954377, 0.1309625291794035, 0.14716040743971653],
        "hard_mask_moving_window_frame_mean_gt": [0.14304103712954377, 0.1309625291794035, 0.14716040743971653]
    }
}

HIGH_LEVEL_COMMAND_LABEL = {
    "flat_left_half": 0,
    "flat_right_half": 1,
    "wave_left_half": 2,
    "wave_right_half": 3,
    "flat_none": 4,
    "wave_none": 4
}

STATE_VARS_POS = ["position_x", "position_y", "position_z"]
STATE_VARS_VEL = ["velocity_x", "velocity_y", "velocity_z"]
STATE_VARS_ACC = ["acceleration_x", "acceleration_y", "acceleration_z"]
STATE_VARS_ROT = ["rotation_w", "rotation_x", "rotation_y", "rotation_z"]
STATE_VARS_OMEGA = ["omega_x", "omega_y", "omega_z"]
STATE_VARS_SHORTHAND_DICT = {
    "pos": STATE_VARS_POS,
    "vel": STATE_VARS_VEL,
    "acc": STATE_VARS_ACC,
    "rot": STATE_VARS_ROT,
    "omega": STATE_VARS_OMEGA,
}
STATE_VARS_UNIT_SHORTHAND_DICT = {
    "pos": "m",
    "vel": "m/s",
    "acc": "m/s/s",
    "rot": "quaternion",
    "omega": "rad/s",
}
