#include "flightlib/envs/racing_env/racing_env.hpp"

namespace flightlib {

RacingEnv::RacingEnv() : RacingEnv(getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/racing_env.yaml")) {}

RacingEnv::RacingEnv(const std::string &cfg_path, const bool rendering_only) {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);
  loadParam(cfg_);

  Vector<3> scale_vector;
  if (rendering_only) {
    scale_vector = Vector<3>(0.0, 0.0, 0.0);
  } else {
    scale_vector = Vector<3>(0.5, 0.5, 0.5);
  }

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  quadrotor_ptr_->setSize(scale_vector);

  if (!rendering_only) {
    // update dynamics
    QuadrotorDynamics dynamics;
    dynamics.updateParams(cfg_);
    quadrotor_ptr_->updateDynamics(dynamics);
  }

  // update state
  quad_state_.x = Vector<25>::Zero();
  quad_state_.t = (Scalar) 0.0f;

  // define a bounding box
  world_box_ << -100, 100, -100, 100, -100, 100;
  quadrotor_ptr_->setWorldBox(world_box_);

  Matrix<3, 3> R_BC;
  Vector<3> B_r_BC;
  if (rendering_only) {
    B_r_BC = Vector<3>(0.0, 0.0, 0.0);
    R_BC = Quaternion(std::cos(-0.5 * M_PI_2), 0.0, 0.0, std::sin(-0.5 * M_PI_2)).toRotationMatrix();
    // R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  } else {
    // airsim
    B_r_BC = Vector<3>(0.2, 0.0, 0.1);
    // float uptilt_angle = 30.0;
    float uptilt_angle = 25.0;
    uptilt_angle = -(uptilt_angle / 90.0) * M_PI_2;
    R_BC = Quaternion(std::cos(0.5 * uptilt_angle), 0.0, std::sin(0.5 * uptilt_angle), 0.0).toRotationMatrix();
    Matrix<3, 3> temp = Quaternion(std::cos(-0.5 * M_PI_2), 0.0, 0.0, std::sin(-0.5 * M_PI_2)).toRotationMatrix();
    R_BC = R_BC * temp;
  }

  rgb_camera_ = std::make_unique<RGBCamera>();
  /*rgb_camera_->setFOV(racingenv::fov);
  rgb_camera_->setHeight(racingenv::image_height);
  rgb_camera_->setWidth(racingenv::image_width);*/
  rgb_camera_->setFOV(image_fov_);
  rgb_camera_->setHeight(image_height_);
  rgb_camera_->setWidth(image_width_);
  rgb_camera_->setRelPose(B_r_BC, R_BC);
  rgb_camera_->setPostProcesscing(std::vector<bool>{false, false, true});  // optical flow enabled
  quadrotor_ptr_->addRGBCamera(rgb_camera_);

  // add gates, hard-coded for now
  /*
  for (int i = 0; i < racingenv::num_gates; i++) {
    // gates_[i] = std::make_shared<StaticGate>("test_gate_" + std::to_string(i), "rpg_gate");
    // gates_[i]->setPosition(Eigen::Vector3f(POSITIONS[i][0], POSITIONS[i][1], POSITIONS[i][2]));
    // gates_[i]->setQuaternion(Quaternion(std::cos(ORIENTATIONS[i]), 0.0, 0.0, std::sin(ORIENTATIONS[i])));
    // gates_[i]->setSize(Eigen::Vector3f(1.05, 1.05, 1.05));
    int actual_i = (i + 1) % racingenv::num_gates;
    Quaternion tmp(std::cos(ORIENTATIONS[i]), 0.0, 0.0, std::sin(ORIENTATIONS[i]));
    std::cout << actual_i << ": " << tmp.w() << " " << tmp.vec().transpose() << std::endl;
  }
  setWaveTrack(wave_track);
  */

  // std::cout << std::endl << "BREAK" << std::endl << std::endl;

  Quaternion ttmp(std::cos(-0.5 * M_PI_2), 0.0, 0.0, std::sin(-0.5 * M_PI_2));
  for (int i = 0; i < num_gates_; i++) {
    // gates_.push_back(std::make_shared<StaticGate>("racing_gate_" + std::to_string(i), "rpg_gate"));
    // gates_.push_back(std::make_shared<StaticObject>("liftoff_gate_" + std::to_string(i), "liftoff_gate"));
    gates_.push_back(std::make_shared<StaticObject>("liftoff_gate_" + std::to_string(i), gate_types_[i]));
    // gates_[i] = std::make_shared<StaticGate>("test_gate_" + std::to_string(i), "rpg_gate");
    gates_[i]->setPosition(Eigen::Vector3f(gate_positions_[i].data()));
    Quaternion actual(gate_orientations_[i].data());
    // Quaternion actual(gate_orientations_[i][0], gate_orientations_[i][1], gate_orientations_[i][2], gate_orientations_[i][3]);
    actual = actual * ttmp;
    // actual = ttmp * actual;
    gates_[i]->setQuaternion(actual);
    // Eigen::VectorXf test = Eigen::VectorXf::Map(&gate_orientations_[i][0], gate_orientations_[i].size());
    // std::cout << i << ": " << actual.w() << " " << actual.vec().transpose() << std::endl;
    // std::cout << "sanity check: " << gate_orientations_[i][0] << " " << gate_orientations_[i][1] << " " << gate_orientations_[i][2] << " " << gate_orientations_[i][3] << std::endl;
    // gates_[i]->setSize(Eigen::Vector3f(1.05, 1.05, 1.05));
  }

  // add unity
  setUnity(true);
}

RacingEnv::~RacingEnv() {}

/*******************************
 * MAIN METHODS (STEP AND GET) *
 *******************************/

bool RacingEnv::step(const Ref<Vector<>> action) {
  // update command
  cmd_.t += sim_dt_;
  cmd_.collective_thrust = action[0];
  cmd_.omega = action.segment<3>(1);

  // simulate quadrotor
  bool success = quadrotor_ptr_->run(cmd_, sim_dt_);

  // update state
  quadrotor_ptr_->getState(&quad_state_);

  return success;
}

bool RacingEnv::getImage(Ref<ImageFlat<>> image) {
  // std::cout << "C++ test 1" << std::endl;
  if (unity_render_ && unity_ready_) {
    // std::cout << "C++ test 2" << std::endl;
    bool rgb_success = rgb_camera_->getRGBImage(cv_image_);
    // std::cout << "C++ test 3" << std::endl;
    if (rgb_success) {
      // std::cout << "Got RGB successfully" << std::endl;
      ImageChannel<> image_channels_[3];
      // std::cout << "C++ test 3.1 " << " " << cv_image_.rows << " " << cv_image_.cols << std::endl;
      cv::split(cv_image_, cv_channels_);
      // std::cout << "C++ test 4 " << image.size() << " " << image.rows() << " " << image.cols() << std::endl;
      // cv::imwrite("/home/simon/Pictures/test.png", cv_image_);
      for (int i = 0; i < cv_image_.channels(); i++) {
        // std::cout << "C++ test 4.0 " << " " << cv_channels_[i].rows << " " << cv_channels_[i].cols << std::endl;
        // std::cout << "C++ test 4.1 " << image_channels_[i].size() << " " << image_channels_[i].rows() << " " << image_channels_[i].cols() << std::endl;
        cv::cv2eigen(cv_channels_[i], image_channels_[i]);
        Map<ImageFlat<>> image_(image_channels_[i].data(), image_channels_[i].size());
        // std::cout << "C++ test 4.2 " << image_.size() << " " << image_.rows() << " " << image_.cols() << std::endl;
        image.block(i * image_height_ * image_width_, 0, image_height_ * image_width_, 1) = image_;
        // std::cout << "C++ test 4.3" << std::endl;
      }
      // std::cout << "C++ test 5" << std::endl;
    } else {
      // std::cout << "Did not get RGB" << std::endl;
    }
    return rgb_success;
  } else {
    std::cout << "WARNING: Unity rendering not available; cannot get any images." << std::endl;
    return false;
  }

  return true;
}

bool RacingEnv::getOpticalFlow(Ref<ImageFlat<float_t>> optical_flow) {
  if (unity_render_ && unity_ready_) {
    ImageChannel<float_t> optical_flow_channels_[2];
    bool flow_success = rgb_camera_->getOpticalFlow(cv_image_);
    if (flow_success) {
      // std::cout << "Got optical flow successfully" << std::endl;
      cv::split(cv_image_, cv_channels_);
      if (cv_image_.channels() != 2) {
        std::cout << "WARNING: Optical flow returned from Unity does not have the correct number of values." << std::endl;
        return false;
      }

      for (int i = 0; i < cv_image_.channels(); i++) {
        cv::cv2eigen(cv_channels_[i], optical_flow_channels_[i]);
        Map<ImageFlat<float_t>> optical_flow_(optical_flow_channels_[i].data(), optical_flow_channels_[i].size());

        // flow values form Unity need to be multiplied by the image dimensions, since they are given in normalized
        // image coordinates; in addition, the y-values are multiplied by -1.0 to convert from Unity to OpenCV
        float multiplier = image_width_;
        if (i == 1) {
          multiplier = -1.0 * image_height_;
        }
        optical_flow.block(i * image_height_ * image_width_, 0, image_height_ * image_width_, 1) = optical_flow_ * multiplier;
      }
    } else {
      // std::cout << "Did not get optical flow" << std::endl;
    }
    return flow_success;
  } else {
    std::cout << "WARNING: Unity rendering not available; cannot get any images." << std::endl;
    return false;
  }

  return true;
}

void RacingEnv::getState(Ref<Vector<>> state) {
  state.segment<QuadState::IDX::SIZE>(0) = quad_state_.x;
}

/****************************
 * METHODS RELATED TO UNITY *
 ****************************/

bool RacingEnv::render() {
  // std::cout << "Quad state on render:" << std::endl << quad_state_.x.segment(0, 7) << std::endl;
  if (unity_render_ && unity_ready_) {
    bool test_ = false;
    int c_ = 0;
    // unity_bridge_ptr_->getRender(render_counter_);
    do {
      // std::cout << "attempting to render and get output " << c_ << std::endl;
      unity_bridge_ptr_->getRender(render_counter_);
      test_ = unity_bridge_ptr_->handleOutput();
      // std::cout << "after rendeirng/output " << c_ << ", result is: " << test_ << std::endl;
      c_++;
    } while (!test_);
    render_counter_++;
  } else {
    std::cout << "WARNING: Unity rendering not available; cannot get images." << std::endl;
    return false;
  }
  return true;
}

void RacingEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
  for (int i = 0; i < num_gates_; i++) {
    bridge->addStaticObject(gates_[i]);
  }
}

bool RacingEnv::setUnity(bool render) {
  unity_render_ = render;
  if (unity_render_ && unity_bridge_ptr_ == nullptr) {
    // create unity bridge
    unity_bridge_ptr_ = UnityBridge::getInstance();
    // add this environment to Unity
    this->addObjectsToUnity(unity_bridge_ptr_);
  }
  return true;
}

bool RacingEnv::connectUnity(const int pub_port, const int sub_port) {
  if (unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_, pub_port, sub_port);
  return unity_ready_;
}

void RacingEnv::disconnectUnity(void) {
  render_counter_ = 0;
  if (unity_bridge_ptr_ != nullptr) {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  } else {
    std::cout << "WARNING: Flightmare Unity Bridge is not initialized." << std::endl;
  }
}

/************************
 * OTHER SETTER METHODS *
 ************************/

void RacingEnv::setReducedState(const Ref<Vector<>> new_state, const int num_vars) {
  quad_state_.x.segment(0, num_vars) = new_state.segment(0, num_vars);  // should maybe express this as sum instead of fixed number
  quadrotor_ptr_->setState(quad_state_);
}

void RacingEnv::setWaveTrack(bool wave_track) {
  float pos_z;
  int i;
  for (int j = 0; j < racingenv::num_elevated_gates; j++) {
    i = ELEVATED_GATES_INDICES[j];
    pos_z = POSITIONS[i][2];
    if (wave_track) {
      pos_z += 3.0;
    }
    // std::cout << pos_z << std::endl;
    gates_[i]->setPosition(Eigen::Vector3f(POSITIONS[i][0], POSITIONS[i][1], pos_z));
  }
}

bool RacingEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["camera"]) {
    image_height_ = cfg["camera"]["height"].as<int>();
    image_width_ = cfg["camera"]["width"].as<int>();
    image_fov_ = cfg["camera"]["fov"].as<Scalar>();
  }

  if (cfg["track"]) {
    if (!(cfg["track"]["positions"] && cfg["track"]["orientations"])) {
      std::cout << "WARNING: Both positions and orientations have to be provided for building a track." << std::endl;
      return false;
    }

    gate_positions_ = cfg["track"]["positions"].as<std::vector<std::vector<Scalar>>>();
    gate_orientations_ = cfg["track"]["orientations"].as<std::vector<std::vector<Scalar>>>();
    gate_types_ = cfg["track"]["types"].as<std::vector<std::string>>();

    num_gates_ = gate_positions_.size();
    if (num_gates_ != gate_orientations_.size()) {
      std::cout << "WARNING: Length of provided gate positions and orientations does not match." << std::endl;
      return false;
    } else if (num_gates_ != gate_types_.size()) {
      std::cout << "WARNING: Length of provided gate positions and types does not match." << std::endl;
      return false;
    }

    /*
    Eigen::MatrixXf test(3, 3);
    for (int i = 0; i < 3; i++) {
      test.row(i) = Eigen::VectorXf::Map(&test_yaml_[i][0], test_yaml_[i].size());
    }
    */

  } else {
    std::cout << "WARNING: No track configuration provided for Racing Environment." << std::endl;
    return false;
  }

  return true;
}

}  // namespace flightlib
