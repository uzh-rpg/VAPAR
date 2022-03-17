#include "flightlib/bridges/unity_bridge.hpp"

namespace flightlib {

// constructor
UnityBridge::UnityBridge()
  : client_address_("tcp://*"),
    pub_port_("10253"),
    sub_port_("10254"),
    num_frames_(0),
    last_downloaded_utime_(0),
    last_download_debug_utime_(0),
    u_packet_latency_(0),
    unity_ready_(false) {
  // initialize connections upon creating unity bridge
  // initializeConnections();
}

bool UnityBridge::initializeConnections() {
  logger_.info("Initializing ZMQ connection!");

  // create and bind an upload socket
  pub_.set(zmqpp::socket_option::send_high_water_mark, 6);
  pub_.bind(client_address_ + ":" + pub_port_);

  // create and bind a download_socket
  sub_.set(zmqpp::socket_option::receive_high_water_mark, 6);
  sub_.set(zmqpp::socket_option::receive_timeout, 1000);
  sub_.bind(client_address_ + ":" + sub_port_);

  // subscribe all messages from ZMQ
  sub_.subscribe("");

  logger_.info("Initializing ZMQ connections done!");
  return true;
}

bool UnityBridge::connectUnity(const SceneID scene_id, const int pub_port, const int sub_port) {
  pub_port_ = std::to_string(pub_port);
  sub_port_ = std::to_string(sub_port);

  // initialize connections
  if (!connections_initialized_) {
    initializeConnections();
    connections_initialized_ = true;
  }

  Scalar time_out_count = 0;
  Scalar sleep_useconds = 0.2 * 1e5;
  setScene(scene_id);
  // try to connect unity
  logger_.info("Trying to Connect Unity.");
  std::cout << "[";
  while (!unity_ready_) {
    // if time out
    if (time_out_count / 1e6 > unity_connection_time_out_) {
      std::cout << "]" << std::endl;
      logger_.warn(
        "Unity Connection time out! Make sure that Unity Standalone "
        "or Unity Editor is running the Flightmare.");
      return false;
    }
    // initialize Scene settings
    sendInitialSettings();
    // check if setting is done
    unity_ready_ = handleSettings();
    // sleep
    usleep(sleep_useconds);
    // increase time out counter
    time_out_count += sleep_useconds;
    // print something
    std::cout << ".";
    std::cout.flush();
  }
  logger_.info("Flightmare Unity is connected.");
  return unity_ready_;
}

bool UnityBridge::disconnectUnity() {
  unity_ready_ = false;
  // create new message object
  pub_.close();
  sub_.close();

  connections_initialized_ = false;
  pub_ = zmqpp::socket(context_, zmqpp::socket_type::publish);
  sub_ = zmqpp::socket(context_, zmqpp::socket_type::subscribe);

  return true;
}

bool UnityBridge::sendInitialSettings(void) {
  // create new message object
  zmqpp::message msg;
  // add topic header
  msg << "Pose";
  // create JSON object for initial settings
  json json_mesg = settings_;
  msg << json_mesg.dump();
  // send message without blocking
  pub_.send(msg, true);
  return true;
};

bool UnityBridge::handleSettings(void) {
  // create new message object
  zmqpp::message msg;

  bool done = false;
  // Unpack message metadata
  if (sub_.receive(msg, true)) {
    std::string metadata_string = msg.get(0);
    // Parse metadata
    if (json::parse(metadata_string).size() > 1) {
      return false;  // hack
    }
    done = json::parse(metadata_string).at("ready").get<bool>();
  }
  return done;
};

bool UnityBridge::getRender(const FrameID frame_id) {
  pub_msg_.frame_id = frame_id;
  QuadState quad_state;
  for (size_t idx = 0; idx < pub_msg_.vehicles.size(); idx++) {
    unity_quadrotors_[idx]->getState(&quad_state);
    pub_msg_.vehicles[idx].position = positionRos2Unity(quad_state.p);
    pub_msg_.vehicles[idx].rotation = quaternionRos2Unity(quad_state.q());
    /*
    std::cout << "Quad position on render request (Flightmare):" << std::endl << quad_state.p << std::endl;
    std::cout << "Quad position on render request (Unity):" << std::endl;
    for (int i = 0; i < pub_msg_.vehicles[idx].position.size(); i++)
      std::cout << pub_msg_.vehicles[idx].position[i] << std::endl;
    std::cout << "Quad rotation on render request (Flightmare):" << std::endl << quad_state.q().coeffs() << std::endl;
    std::cout << "Quad rotation on render request (Unity):" << std::endl;
    for (int i = 0; i < pub_msg_.vehicles[idx].rotation.size(); i++)
      std::cout << pub_msg_.vehicles[idx].rotation[i] << std::endl;
    */
  }

  for (size_t idx = 0; idx < pub_msg_.objects.size(); idx++) {
    std::shared_ptr<StaticObject> gate = static_objects_[idx];
    pub_msg_.objects[idx].position = positionRos2Unity(gate->getPosition());
    pub_msg_.objects[idx].rotation = quaternionRos2Unity(gate->getQuaternion());
  }

  // std::cout << "sending rendering request with frame ID " << pub_msg_.frame_id << std::endl;

  // create new message object
  zmqpp::message msg;
  // add topic header
  msg << "Pose";
  // create JSON object for pose update and append
  json json_msg = pub_msg_;
  msg << json_msg.dump();
  // send message without blocking
  pub_.send(msg, true);
  return true;
}

bool UnityBridge::setScene(const SceneID& scene_id) {
  if (scene_id >= UnityScene::SceneNum) {
    logger_.warn("Scene ID is not defined, cannot set scene.");
    return false;
  }
  // logger_.info("Scene ID is set to %d.", scene_id);
  settings_.scene_id = scene_id;
  return true;
}

/*
bool UnityBridge::setCameraParameters(Scalar fov, int width, int height) {
  for (size_t idx = 0; idx < pub_msg_.cameras.size(); idx++) {
    pub_msg_.cameras[idx].fov = fov;
    pub_msg_.cameras[idx].width = width;
    pub_msg_.cameras[idx].height = height;
  }
  return true;
}
*/

bool UnityBridge::addQuadrotor(std::shared_ptr<Quadrotor> quad) {
  Vehicle_t vehicle_t;
  // get quadrotor state
  QuadState quad_state;
  if (!quad->getState(&quad_state)) {
    logger_.error("Cannot get Quadrotor state.");
    return false;
  }

  vehicle_t.ID = "quadrotor" + std::to_string(settings_.vehicles.size());
  vehicle_t.position = positionRos2Unity(quad_state.p);
  vehicle_t.rotation = quaternionRos2Unity(quad_state.q());
  vehicle_t.size = scalarRos2Unity(quad->getSize());

  // get camera
  std::vector<std::shared_ptr<RGBCamera>> rgb_cameras = quad->getCameras();
  for (size_t cam_idx = 0; cam_idx < rgb_cameras.size(); cam_idx++) {
    std::shared_ptr<RGBCamera> cam = rgb_cameras[cam_idx];
    Camera_t camera_t;
    camera_t.ID = vehicle_t.ID + "_" + std::to_string(cam_idx);
    camera_t.T_BC = transformationRos2Unity(rgb_cameras[cam_idx]->getRelPose());
    camera_t.channels = rgb_cameras[cam_idx]->getChannels();
    camera_t.width = rgb_cameras[cam_idx]->getWidth();
    camera_t.height = rgb_cameras[cam_idx]->getHeight();
    camera_t.fov = rgb_cameras[cam_idx]->getFOV();
    camera_t.depth_scale = rgb_cameras[cam_idx]->getDepthScale();
    camera_t.enabled_layers = rgb_cameras[cam_idx]->getEnabledLayers();
    camera_t.is_depth = false;
    camera_t.output_index = cam_idx;
    vehicle_t.cameras.push_back(camera_t);

    // add rgb_cameras
    rgb_cameras_.push_back(rgb_cameras[cam_idx]);
  }
  unity_quadrotors_.push_back(quad);

  //
  settings_.vehicles.push_back(vehicle_t);
  pub_msg_.vehicles.push_back(vehicle_t);
  return true;
}

bool UnityBridge::addStaticObject(std::shared_ptr<StaticObject> static_object) {
  Object_t object_t;
  object_t.ID = static_object->getID();
  object_t.prefab_ID = static_object->getPrefabID();
  object_t.position = positionRos2Unity(static_object->getPosition());
  object_t.rotation = quaternionRos2Unity(static_object->getQuaternion());
  object_t.size = scalarRos2Unity(static_object->getSize());

  static_objects_.push_back(static_object);
  settings_.objects.push_back(object_t);
  pub_msg_.objects.push_back(object_t);
  //
  return true;
}

bool UnityBridge::handleOutput() {
  // create new message object
  zmqpp::message msg;
  SubMessage_t sub_msg;

  int received_frame_id = -1;

  // std::cout << "listening for response" << std::endl;

  while (received_frame_id != pub_msg_.frame_id) {
      // bool got_message = sub_.receive(msg, true);
      bool got_message = sub_.receive(msg);
      if (!got_message) {
        return false;
      }
      //if (got_message) {
        // unpack message metadata
        std::string json_sub_msg = msg.get(0);
        // parse metadata
        sub_msg = json::parse(json_sub_msg);
        received_frame_id = sub_msg.frame_id;

        // std::cout << "received response with frame ID " << received_frame_id << " (expected " << pub_msg_.frame_id << ")" << std::endl;
      //} else {
      //  break;
      //}
  }

  if (received_frame_id == -1) {
    // std::cout << "failed to receive rendering output" << std::endl;
    return false;
  }

  // std::cout << "Current sent frame ID: " << pub_msg_.frame_id << std::endl;
  // std::cout << "Current received frame ID: " << sub_msg.frame_id << std::endl;

  // std::cout << "handle output 0" << std::endl;


  size_t image_i = 1;
  // ensureBufferIsAllocated(sub_msg);
  for (size_t idx = 0; idx < settings_.vehicles.size(); idx++) {
    // update vehicle collision flag
    unity_quadrotors_[idx]->setCollision(sub_msg.sub_vehicles[idx].collision);

    // std::cout << "vehicle start " << idx << std::endl;
    int cam_idx = 0;

    // feed image data to RGB camera
    for (const auto& cam : settings_.vehicles[idx].cameras) {

      // std::cout << "cam start " << cam_idx << std::endl;
      cam_idx++;
      for (size_t layer_idx = 0; layer_idx <= cam.enabled_layers.size();
           layer_idx++) {

        //std::cout << "layer start " << layer_idx << std::endl;
        if (!layer_idx == 0 && !cam.enabled_layers[layer_idx - 1]) continue;
        //std::cout << "layer used " << layer_idx << std::endl;

        if (layer_idx == 1) {
          // depth
          uint32_t image_len = cam.width * cam.height * 4;
          // Get raw image bytes from ZMQ message.
          // WARNING: This is a zero-copy operation that also casts the input to
          // an array of unit8_t. when the message is deleted, this pointer is
          // also dereferenced.
          const uint8_t* image_data;
          msg.get(image_data, image_i);
          image_i = image_i + 1;
          // Pack image into cv::Mat
          cv::Mat new_image = cv::Mat(cam.height, cam.width, CV_32FC1);
          memcpy(new_image.data, image_data, image_len);
          // Flip image since OpenCV origin is upper left, but Unity's is lower
          // left.
          new_image = new_image * (100.f);
          cv::flip(new_image, new_image, 0);


          unity_quadrotors_[idx]
            ->getCameras()[cam.output_index]
            ->feedImageQueue(layer_idx, new_image);


        } else if (layer_idx == 3) {
            //std::cout << "handle output 1" << std::endl;
            // optiindexing c++cal flow
            uint32_t image_len = cam.width * cam.height * sizeof(float_t) * 2;
            // doesn't really matter which type we take, but since we wrote "raw bytes"
            // in Unity, we might as well read them here as well
            const uint8_t* image_data;

            //std::cout << "handle output 2" << std::endl;

            // just get the raw binary data stored in the message (I guess at index image_i?)
            msg.get(image_data, image_i);

            //std::cout << "handle output 3" << std::endl;

            image_i = image_i + 1;

            // Pack image into cv::Mat
            // cv::Mat new_image = cv::Mat(cam.height, cam.width, CV_MAKETYPE(CV_32F, 2));
            cv::Mat new_image = cv::Mat(cam.height, cam.width, CV_32FC2);
            //std::cout << "handle output 4" << std::endl;

            // copy the raw data directly over from the "buffer" to the OpenCV matrix
            memcpy(new_image.data, image_data, image_len);

            //std::cout << "handle output 5" << std::endl;

            // flip from Unity to OpenCV coordinates
            cv::flip(new_image, new_image, 0);

            //std::cout << "handle output 6" << std::endl;


            unity_quadrotors_[idx]
                ->getCameras()[cam.output_index]
                ->feedImageQueue(layer_idx, new_image);

            //std::cout << "handle output 7" << std::endl;
        } else {
          uint32_t image_len = cam.width * cam.height * cam.channels;
          // Get raw image bytes from ZMQ message.
          // WARNING: This is a zero-copy operation that also casts the input to
          // an array of unit8_t. when the message is deleted, this pointer is
          // also dereferenced.
          const uint8_t* image_data;
          msg.get(image_data, image_i);
          image_i = image_i + 1;
          // Pack image into cv::Mat
          cv::Mat new_image =
            cv::Mat(cam.height, cam.width, CV_MAKETYPE(CV_8U, cam.channels));
          memcpy(new_image.data, image_data, image_len);
          // Flip image since OpenCV origin is upper left, but Unity's is lower
          // left.
          cv::flip(new_image, new_image, 0);

          // Tell OpenCv that the input is RGB.
          if (cam.channels == 3) {
            cv::cvtColor(new_image, new_image, CV_RGB2BGR);
            // cv::imwrite("/home/simon/Documents/Meetings/week_25/debug_flightmare/flightmare/" + std::to_string(received_frame_id) + ".png", new_image);
          }
          unity_quadrotors_[idx]
            ->getCameras()[cam.output_index]
            ->feedImageQueue(layer_idx, new_image);
        }
      }
    }
  }
  //std::cout << "handle output 8" << std::endl;
  return true;
}

bool UnityBridge::getPointCloud(PointCloudMessage_t& pointcloud_msg,
                                Scalar time_out) {
  // create new message object
  zmqpp::message msg;
  // add topic header
  msg << "PointCloud";
  // create JSON object for initial settings
  json json_msg = pointcloud_msg;
  msg << json_msg.dump();
  // send message without blocking
  pub_.send(msg, true);

  std::cout << "Generate PointCloud: Timeout=" << (int)time_out << " seconds."
            << std::endl;

  Scalar run_time = 0.0;
  while (!std::experimental::filesystem::exists(
    pointcloud_msg.path + pointcloud_msg.file_name + ".ply")) {
    if (run_time >= time_out) {
      logger_.warn("Timeout... PointCloud was not saved within expected time.");
      return false;
    }
    std::cout << "Waiting for Pointcloud: Current Runtime=" << (int)run_time
              << " seconds." << std::endl;
    usleep((time_out / 10.0) * 1e6);
    run_time += time_out / 10.0;
  }
  return true;
}

}  // namespace flightlib
