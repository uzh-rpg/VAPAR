#pragma once

// standard library
#include <unistd.h>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

// yaml
#include <yaml-cpp/yaml.h>

// Flightmare stuff
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

namespace flightlib {

class EnvBaseCamera {
 public:
  EnvBaseCamera();
  virtual ~EnvBaseCamera() = 0;

  // (pure virtual) public methods (have to be implemented by child classes)
  virtual bool step(const Ref<Vector<>> action) = 0;
  virtual bool getImage(Ref<ImageFlat<>> image) = 0;
  virtual bool getOpticalFlow(Ref<ImageFlat<float_t>> optical_flow) = 0;
  virtual void getState(Ref<Vector<>> state) = 0;

  // Unity methods
  virtual bool render() = 0;
  virtual void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) = 0;
  virtual bool setUnity(bool render) = 0;
  virtual bool connectUnity(const int pub_port = 10253, const int sub_port = 10254) = 0;
  virtual void disconnectUnity() = 0;

  // auxiliary functions
  inline int getImageHeight() { return image_height_; };
  inline int getImageWidth() {return image_width_; };
  inline int getStateDim() { return QuadState::IDX::SIZE; }
  inline Scalar getSimTimeStep() { return sim_dt_; };

  inline void setSimTimeStep(Scalar time_step) { sim_dt_ = time_step; };
  inline void setSceneID(SceneID scene_id) { scene_id_ = scene_id; };

 protected:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Matrix<3, 2> world_box_;

  // control time step
  Scalar sim_dt_{0.02};

  // camera
  int image_height_ = 600, image_width_ = 800;
  Scalar image_fov_ = 90.0;
  std::shared_ptr<RGBCamera> rgb_camera_;

  // image observations
  cv::Mat cv_image_;
  cv::Mat cv_channels_[3];

  // needed to sync outgoing requests for frames and incoming "results"/renders
  unsigned long render_counter_{0};

  // unity
  std::shared_ptr<UnityBridge> unity_bridge_ptr_;
  // SceneID scene_id_{UnityScene::Stadium};
  SceneID scene_id_{UnityScene::STADIUM};
  bool unity_ready_{false};
  bool unity_render_{false};
};

}  // namespace flightlib