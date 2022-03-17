/*#include "flightlib/envs/racing_env/racing_test_env.hpp"

namespace flightlib {

RacingTestEnv::RacingTestEnv()
  : RacingTestEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/racing_test_env.yaml")) {}

RacingTestEnv::RacingTestEnv(const std::string &cfg_path)
  : EnvBaseCamera() {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  // load parameters
  loadParam(cfg_);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // define a bounding box
  world_box_ << -30, 30, -30, 30, 0, 30;
  quadrotor_ptr_->setWorldBox(world_box_);

  // define input and output dimension for the environment
  obs_dim_ = racingtestenv::kNObs;
  act_dim_ = racingtestenv::kNAct;

  // add camera
  rgb_camera_ = std::make_unique<RGBCamera>();
  image_counter_ = 0;
  Vector<3> B_r_BC(0.0, -0.5, 0.3);
  Matrix<3, 3> R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  rgb_camera_->setFOV(racingtestenv::fov);
  rgb_camera_->setHeight(racingtestenv::image_height);
  rgb_camera_->setWidth(racingtestenv::image_width);
  rgb_camera_->setRelPose(B_r_BC, R_BC);
  rgb_camera_->setPostProcesscing(std::vector<bool>{false, false, false});
  quadrotor_ptr_->addRGBCamera(rgb_camera_);

  // add gates, hard-coded for now
  float positions[racingtestenv::num_gates][3] = {
    {-18.0,  10.0, 2.5},
    {-25.0,   0.0, 2.5},
    {-18.0, -10.0, 2.5},
    { -1.3,  -1.3, 2.5},
    {  1.3,   1.3, 2.5},
    { 18.0,  10.0, 2.5},
    { 25.0,   0.0, 2.5},
    { 18.0, -10.0, 2.5},
    {  1.3,  -1.3, 2.5},
    { -1.3,   1.3, 2.5},
  };
  // only need the rotation angle around the z-axis
  float orientations[racingtestenv::num_gates] = {
    -0.75 * M_PI_2,
    -0.50 * M_PI_2,
    -0.25 * M_PI_2,
     0.25 * M_PI_2,
     0.25 * M_PI_2,
    -0.25 * M_PI_2,
    -0.50 * M_PI_2,
    -0.75 * M_PI_2,
     0.75 * M_PI_2,
     0.75 * M_PI_2,
  };
  for (int i = 0; i < racingtestenv::num_gates; i++) {
    gates_[i] = std::make_shared<StaticGate>("test_gate_" + std::to_string(i), "rpg_gate");
    gates_[i]->setPosition(Eigen::Vector3f(positions[i][1], positions[i][0], positions[i][2]));
    gates_[i]->setRotation(Quaternion(std::cos(-orientations[i]), 0.0, 0.0, std::sin(-orientations[i])));
  }

  // add unity
  setUnity(true);

  Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<racingtestenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<racingtestenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;
}

RacingTestEnv::~RacingTestEnv() {}

/*
void RacingTestEnv::test(Ref<ImageFlat<>> image_test) {
  // taken from https://stackoverflow.com/a/45057328

  cv::split(cv_image_, cv_channels_);
  for (int i = 0; i < cv_image_.channels(); i++) {
    cv::cv2eigen(cv_channels_[i], channels_[i]);
    Map<ImageFlat<>> image_(channels_[i].data(), channels_[i].size());
    image_test.block<racingtestenv::image_height * racingtestenv::image_width, 1>(i * racingtestenv::image_height * racingtestenv::image_width, 0) = image_;
  }

  /*
  constexpr uint32_t height = 3;
  constexpr uint32_t width = 7;

  cv::Mat img(height, width, CV_32FC3, cv::Scalar(1.0f, 2.0f, 3.0f));

  using MatrixXfRowMajor = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using C3Stride = Eigen::Stride<Eigen::Dynamic, 3>;
  C3Stride c3Stride(width * 3, 3);

  using cvMap = Eigen::Map<MatrixXfRowMajor, Eigen::Unaligned, C3Stride>;
  cvMap imgC1(reinterpret_cast<Scalar*>(img.data) + 0, img.rows, img.cols, c3Stride);
  cvMap imgC2(reinterpret_cast<Scalar*>(img.data) + 1, img.rows, img.cols, c3Stride);
  cvMap imgC3(reinterpret_cast<Scalar*>(img.data) + 2, img.rows, img.cols, c3Stride);

  std::cout << "'Image' channels:" << std::endl;
  std::cout << imgC1 << std::endl << std::endl;
  std::cout << imgC2 << std::endl << std::endl;
  std::cout << imgC3 << std::endl << std::endl;

  channel_test = imgC1;
}
*//*

bool RacingTestEnv::reset(Ref<Vector<>> obs, Ref<ImageFlat<>> image, const bool random) {
  image_counter_ = 0;
  quad_state_.setZero();
  quad_act_.setZero();

  if (random) {
    // randomly reset the quadrotor state
    // reset position
    quad_state_.x(QS::POSX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSZ) = uniform_dist_(random_gen_) + 5;
    if (quad_state_.x(QS::POSZ) < -0.0)
      quad_state_.x(QS::POSZ) = -quad_state_.x(QS::POSZ);
    // reset linear velocity
    quad_state_.x(QS::VELX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELZ) = uniform_dist_(random_gen_);
    // reset orientation
    quad_state_.x(QS::ATTW) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_);
    quad_state_.qx /= quad_state_.qx.norm();
  }
  // reset quadrotor with random states
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  cmd_.thrusts.setZero();

  // obtain observations
  getObs(obs, image);
  return true;
}

bool RacingTestEnv::getObs(Ref<Vector<>> obs, Ref<ImageFlat<>> image) {
  quadrotor_ptr_->getState(&quad_state_);

  // convert quaternion to euler angle
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
  // quaternionToEuler(quad_state_.q(), euler);
  quad_obs_ << quad_state_.p, euler_zyx, quad_state_.v, quad_state_.w;

  // see here: https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
  // apparently this just means that we assign to a segment of kNObs entries, starting at position
  //  kObs of the vector obs (in this case this just seems to be the start of the vector)
  obs.segment<racingtestenv::kNObs>(racingtestenv::kObs) = quad_obs_;

  // also capture an image
  // for now just print some information, since I don't know how the conversion from C++ to numpy should work
  if (unity_render_ && unity_ready_) {
    unity_bridge_ptr_->getRender(0);
    unity_bridge_ptr_->handleOutput();

    bool rgb_success = rgb_camera_->getRGBImage(cv_image_);

    if (rgb_success) {
      cv::split(cv_image_, cv_channels_);
      for (int i = 0; i < cv_image_.channels(); i++) {
        cv::cv2eigen(cv_channels_[i], channels_[i]);
        Map<ImageFlat<>> image_(channels_[i].data(), channels_[i].size());
        image.block<racingtestenv::image_height * racingtestenv::image_width, 1>(i * racingtestenv::image_height * racingtestenv::image_width, 0) = image_;
      }
    }
    /*
    std::cout << image << std::endl;
    Eigen::Matrix<uint8_t, 10, 1> test = image.block(1000, 0, 1010, 1);
    std::cout << "C++: " << std::endl;
    for (int i = 0; i < 10; i++) {
      std::cout << image[1000 + i] << std::endl;
    }
    // std::cout << "image data type: " << this->type2str(image_.type()) << std::endl;
    // std::cout << "CAMERA IMAGE" << std::endl;
    // std::cout << "success: " << rgb_success << ", rows: " << cv_image_.rows << ", cols: " << cv_image_.cols << std::endl;
    if (rgb_success) {
      cv::imwrite("/home/simon/Desktop/flightmare_cam_test/" + std::to_string(image_counter_) + ".png", cv_image_);
    }
    image_counter_++;*//*
  } else {
    std::cout << "Unity rendering not available; cannot get images." << std::endl;
  }

  return true;
}

Scalar RacingTestEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs, Ref<ImageFlat<>> image) {
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.thrusts = quad_act_;

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(obs, image);

  return 0.0;
}

bool RacingTestEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["racing_test_env"]) {
    sim_dt_ = cfg["racing_test_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["racing_test_env"]["max_t"].as<Scalar>();
  } else {
    return false;
  }

  if (cfg["gate_position"]) {
    x_ = cfg["gate_position"]["x"].as<Scalar>();
    y_ = cfg["gate_position"]["y"].as<Scalar>();
    z_ = cfg["gate_position"]["z"].as<Scalar>();
  } else {
    return false;
  }

  if (cfg["gate_orientation"]) {
    or_w_ = cfg["gate_orientation"]["w"].as<Scalar>();
    or_x_ = cfg["gate_orientation"]["x"].as<Scalar>();
    or_y_ = cfg["gate_orientation"]["y"].as<Scalar>();
    or_z_ = cfg["gate_orientation"]["z"].as<Scalar>();
  } else {
    return false;
  }

  return true;
}

bool RacingTestEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool RacingTestEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

int RacingTestEnv::getImageHeight() const {
  return racingtestenv::image_height;
}

int RacingTestEnv::getImageWidth() const {
  return racingtestenv::image_width;
}

void RacingTestEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
  //bridge->addStaticObject(gate_);
  for (int i = 0; i < racingtestenv::num_gates; i++) {
    bridge->addStaticObject(gates_[i]);
  }
}

bool RacingTestEnv::setUnity(bool render) {
  unity_render_ = render;
  if (unity_render_ && unity_bridge_ptr_ == nullptr) {
    // create unity bridge
    unity_bridge_ptr_ = UnityBridge::getInstance();
    // add this environment to Unity
    this->addObjectsToUnity(unity_bridge_ptr_);
    logger_.info("Flightmare Bridge is created.");
  }
  return true;
}

bool RacingTestEnv::connectUnity(void) {
  if (unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}

void RacingTestEnv::disconnectUnity(void) {
  if (unity_bridge_ptr_ != nullptr) {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  } else {
    logger_.warn("Flightmare Unity Bridge is not initialized.");
  }
}

std::ostream &operator<<(std::ostream &os, const RacingTestEnv &quad_env) {
  os.precision(3);
  os << "Racing Test Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib
*/