import datetime
import os
import shutil

import yaml


def create_settings(settings_yaml, mode='test', generate_log=True):
    setting_dict = {'train': TrainSetting,
                    'test': TestSetting,
                    'dagger': DaggerSetting}
    settings = setting_dict.get(mode, None)
    if settings is None:
        raise IOError("Unidentified Settings")
    settings = settings(settings_yaml, generate_log)
    return settings


class Settings:

    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)

            self.quad_name = settings['quad_name']

            self.seq_len = settings['seq_len']

            self.env_config_path = os.path.join(
                os.getenv("FLIGHTMARE_PATH"),
                "flightlib/configs",
                "{}.yaml".format(settings.get("env_config", "racing_env")),
            )

            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.resume_ckpt_file = checkpoint['resume_file']

            # Save a copy of the parameters for reproducibility
            log_root = settings['log_dir']
            if not log_root == '' and generate_log:
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                if os.path.isabs(log_root):
                    self.log_dir = os.path.join(log_root, current_time)
                else:
                    self.log_dir = os.path.join(os.getenv("DDA_ROOT"), log_root, current_time)
                os.makedirs(self.log_dir)
                net_file = os.path.join(os.getenv("FLIGHTMARE_PATH"),
                                        "flightil/dda/models/nets.py")
                assert os.path.isfile(net_file)
                shutil.copy(net_file, self.log_dir)
                shutil.copy(settings_yaml, self.log_dir)

    def add_flags(self):
        self._add_flags()

    def _add_flags(self):
        raise NotImplementedError


class TrainSetting(Settings):

    def __init__(self, settings_yaml, generate_log=True):
        super(TrainSetting, self).__init__(settings_yaml, generate_log=generate_log)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def _add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            # --- Train Time --- #
            train_conf = settings['train']
            self.gpu = train_conf["gpu"]
            self.max_training_epochs = train_conf['max_training_epochs']
            self.max_allowed_error = train_conf['max_allowed_error']
            self.batch_size = train_conf['batch_size']
            self.learning_rate = train_conf["learning_rate"]
            self.summary_freq = train_conf['summary_freq']
            if os.path.isabs(train_conf["train_dir"]):
                self.train_dir = train_conf["train_dir"]
            else:
                self.train_dir = os.path.join(os.getenv("DDA_ROOT"), train_conf["train_dir"])
            if os.path.isabs(train_conf["val_dir"]):
                self.val_dir = train_conf["val_dir"]
            else:
                self.val_dir = os.path.join(os.getenv("DDA_ROOT"), os.pardir, train_conf["val_dir"])
            self.use_fts_tracks = train_conf['use_fts_tracks']
            self.use_images = train_conf.get("use_images", False)
            self.use_imu = train_conf['use_imu']
            self.use_raw_imu_data = train_conf.get("use_raw_imu_data", False)
            self.use_pos = train_conf["use_pos"]
            self.use_activation = train_conf["use_activation"]
            self.imu_no_rot = True if self.use_raw_imu_data else train_conf.get("imu_no_rot", False)
            self.imu_no_vels = train_conf.get("imu_no_vels", False)
            self.no_ref = train_conf.get("no_ref", False)
            self.attention_fts_type = train_conf.get("attention_fts_type", "none")
            self.attention_model_path = train_conf.get("attention_model_path", "")
            self.attention_branching = train_conf.get("attention_branching", False)
            self.attention_branching_threshold = train_conf.get("attention_branching_threshold", 25)
            self.attention_masking = train_conf.get("attention_masking", False) and self.use_images
            self.gate_direction_branching = train_conf.get("gate_direction_branching", False)
            self.gate_direction_branching_threshold = train_conf.get("gate_direction_branching_threshold", 25)
            self.gate_direction_start_gate = train_conf.get("gate_direction_start_gate", 9)
            self.shallow_control_module = train_conf.get("shallow_control_module", False)
            self.min_number_fts = train_conf['min_number_fts']
            self.save_every_n_epochs = train_conf['save_every_n_epochs']

            assert not (self.use_fts_tracks and self.use_images), "Can only use one of feature tracks and images!"
            assert not (self.attention_branching and self.gate_direction_branching), \
                "Can only use one of attention branching and gate direction branching!"


class TestSetting(Settings):

    def __init__(self, settings_yaml, generate_log=True):
        super(TestSetting, self).__init__(settings_yaml, generate_log=generate_log)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def _add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            test_time = settings['test_time']
            self.execute_nw_predictions = test_time['execute_nw_predictions']
            assert isinstance(self.execute_nw_predictions, bool)
            self.max_rollouts = test_time['max_rollouts']
            self.fallback_threshold_rates = test_time['fallback_threshold_rates']
            self.fallback_threshold_thrust = test_time['fallback_threshold_thrust']
            self.min_number_fts = test_time['min_number_fts']
            self.use_imu = test_time['use_imu']
            self.use_fts_tracks = test_time['use_fts_tracks']
            self.verbose = settings['verbose']


class DaggerSetting(Settings):

    def __init__(self, settings_yaml, generate_log=True):
        super(DaggerSetting, self).__init__(settings_yaml, generate_log=generate_log)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def _add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            # --- Data Generation --- #
            data_gen = settings['data_generation']
            self.max_rollouts = data_gen['max_rollouts']
            self.double_th_every_n_rollouts = data_gen['double_th_every_n_rollouts']
            self.train_every_n_rollouts = data_gen['train_every_n_rollouts']
            # --- Test Time --- #
            test_time = settings['test_time']
            self.test_every_n_rollouts = test_time["test_every_n_rollouts"]
            self.execute_nw_predictions = test_time['execute_nw_predictions']
            assert isinstance(self.execute_nw_predictions, bool)
            self.fallback_threshold_rates = test_time['fallback_threshold_rates']
            self.rand_thrust_mag = test_time['rand_thrust_mag']
            self.rand_rate_mag = test_time['rand_rate_mag']
            self.rand_controller_prob = test_time['rand_controller_prob']
            # --- Train Time --- #
            train_conf = settings['train']
            self.gpu = train_conf["gpu"]
            self.max_training_epochs = train_conf['max_training_epochs']
            self.max_allowed_error = train_conf['max_allowed_error']
            self.exclude_collision_rollouts = train_conf.get("exclude_collision_rollouts", True)
            self.batch_size = train_conf['batch_size']
            self.learning_rate = train_conf["learning_rate"]
            self.min_number_fts = train_conf['min_number_fts']
            self.summary_freq = train_conf['summary_freq']
            if os.path.isabs(train_conf["train_dir"]):
                self.train_dir = train_conf["train_dir"]
            else:
                self.train_dir = os.path.join(os.getenv("DDA_ROOT"), train_conf["train_dir"])
            if os.path.isabs(train_conf["val_dir"]):
                self.val_dir = train_conf["val_dir"]
            else:
                self.val_dir = os.path.join(os.getenv("DDA_ROOT"), train_conf["val_dir"])
            self.use_imu = train_conf['use_imu']
            self.use_raw_imu_data = train_conf.get("use_raw_imu_data", False)
            self.use_fts_tracks = train_conf['use_fts_tracks']
            self.use_images = train_conf.get("use_images", False)
            self.use_pos = train_conf["use_pos"]
            self.use_activation = train_conf["use_activation"]
            # TODO: this is very ugly and hacky, but to get things running quickly, I'll do it like this
            self.imu_no_rot = True if self.use_raw_imu_data else train_conf.get("imu_no_rot", False)
            self.imu_no_vels = train_conf.get("imu_no_vels", False)
            self.no_ref = train_conf.get("no_ref", False)
            self.save_every_n_epochs = train_conf['save_every_n_epochs']
            self.verbose = settings['verbose']
            self.attention_fts_type = train_conf.get("attention_fts_type", "none")
            self.attention_model_path = train_conf.get("attention_model_path", "")
            self.attention_record_all_features = train_conf.get("attention_record_all_features", False)
            self.attention_branching = train_conf.get("attention_branching", False)
            self.attention_branching_threshold = train_conf.get("attention_branching_threshold", 25)
            self.attention_masking = train_conf.get("attention_masking", False) and self.use_images
            self.gate_direction_branching = train_conf.get("gate_direction_branching", False)
            self.gate_direction_branching_threshold = train_conf.get("gate_direction_branching_threshold", 25)
            self.gate_direction_start_gate = train_conf.get("gate_direction_start_gate", 9)
            self.save_at_net_frequency = train_conf.get("save_at_net_frequency", False)
            self.shallow_control_module = train_conf.get("shallow_control_module", False)
            self.mpc_time_horizon = train_conf.get("mpc_time_horizon", 2.0)
            self.mpc_time_step = train_conf.get("mpc_time_step", 0.1)
            assert isinstance(self.verbose, bool)
            # --- Flightmare simulation --- #
            sim_conf = settings["simulation"]
            self.flightmare_pub_port = sim_conf["flightmare_pub_port"]
            self.flightmare_sub_port = sim_conf["flightmare_sub_port"]
            self.disconnect_when_training = sim_conf["disconnect_when_training"]
            self.base_frequency = sim_conf["base_frequency"]
            self.image_frequency = sim_conf["image_frequency"]
            self.ref_frequency = sim_conf["ref_frequency"]
            self.command_frequency = sim_conf["command_frequency"]
            self.expert_command_frequency = sim_conf["expert_command_frequency"]
            self.start_buffer = sim_conf["start_buffer"]
            self.max_time = sim_conf["max_time"]
            self.trajectory_path = sim_conf["trajectory_path"]
            if os.path.isdir(self.trajectory_path):
                trajectory_list = []
                for file in os.listdir(self.trajectory_path):
                    trajectory_list.append(os.path.join(self.trajectory_path, file))
                self.trajectory_path = trajectory_list
            self.return_extra_info = sim_conf.get("return_extra_info", False)

            assert not (self.use_fts_tracks and self.use_images), "Can only use one of feature tracks and images!"
            assert not (self.attention_branching and self.gate_direction_branching), \
                "Can only use one of attention branching and gate direction branching!"
