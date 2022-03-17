/*#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <string>

// yaml cpp
#include <yaml-cpp/yaml.h>

// Eigen
#include <eigen3/Eigen/Dense>

// opencv
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base_camera.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/objects/static_gate.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

namespace flightlib {

using ChannelMatrix = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ChannelStride = Eigen::Stride<Eigen::Dynamic, 3>;
template<typename S>
using ChannelMap = Eigen::Map<ChannelMatrix, Eigen::Unaligned, S>;

namespace racingtestenv {

enum Ctl : int {
  // observations
  kObs = 0,
  //
  kPos = 0,
  kNPos = 3,
  kOri = 3,
  kNOri = 3,
  kLinVel = 6,
  kNLinVel = 3,
  kAngVel = 9,
  kNAngVel = 3,
  kNObs = 12,
  // control actions
  kAct = 0,
  kNAct = 4,
  // image dimensions
  image_height = 600,
  image_width = 800,
  fov = 90,
  // track info (should probably be loaded)
  num_gates = 10,
};
};
class RacingTestEnv final : public EnvBaseCamera {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RacingTestEnv();
  RacingTestEnv(const std::string &cfg_path);
  ~RacingTestEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, Ref<ImageFlat<>> image, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs, Ref<ImageFlat<>> image) override;

  // - public set functions
  bool loadParam(const YAML::Node &cfg);

  // - public get functions
  bool getObs(Ref<Vector<>> obs, Ref<ImageFlat<>> image) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;
  int getImageHeight() const;
  int getImageWidth() const;

  // - auxiliary functions
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);
  bool setUnity(bool render);
  bool connectUnity();
  void disconnectUnity();

  friend std::ostream &operator<<(std::ostream &os, const RacingTestEnv &quad_env);

 private:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"RacingTestEnv"};

  // observations and actions (for RL)
  Vector<racingtestenv::kNObs> quad_obs_;
  Vector<racingtestenv::kNAct> quad_act_;

  // camera
  int cam_height_, cam_width_, cam_fov_;
  std::shared_ptr<RGBCamera> rgb_camera_;

  // image observations
  int image_counter_;
  ImageChannel<racingtestenv::image_height, racingtestenv::image_width> channels_[3];
  cv::Mat cv_image_;
  cv::Mat cv_channels_[3];

  // gate(s)
  std::shared_ptr<StaticGate> gates_[10];
  std::shared_ptr<StaticGate> gate_;
  float x_, y_, z_;
  float or_w_, or_x_, or_y_, or_z_;

  // unity
  std::shared_ptr<UnityBridge> unity_bridge_ptr_;
  SceneID scene_id_{UnityScene::WAREHOUSE};
  bool unity_ready_{false};
  bool unity_render_{false};

  // action and observation normalization (for learning)
  Vector<racingtestenv::kNAct> act_mean_;
  Vector<racingtestenv::kNAct> act_std_;
  Vector<racingtestenv::kNObs> obs_mean_ = Vector<racingtestenv::kNObs>::Zero();
  Vector<racingtestenv::kNObs> obs_std_ = Vector<racingtestenv::kNObs>::Ones();

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};

}  // namespace flightlib
*/