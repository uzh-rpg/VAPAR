import os
import numpy as np

from flightgym import RacingEnv


class RacingEnvWrapper:

    def __init__(self, rendering_only=False, config_path=None):
        self.env = RacingEnv(
            config_path or os.path.join(os.getenv("FLIGHTMARE_PATH"), "flightlib/configs/racing_env.yaml"),
            rendering_only,
        )

        self.image_height = self.env.getImageHeight()
        self.image_width = self.env.getImageWidth()
        self.state_dim = self.env.getStateDim()

        self.image = np.zeros((self.image_height * self.image_width * 3,), dtype=np.uint8)
        self.optical_flow = np.zeros((self.image_height * self.image_width * 2,), dtype=np.float32)
        self.state = np.zeros((self.state_dim,), dtype=np.float32)

    def _reshape_image(self, image):
        return np.reshape(image, (3, self.image_height, self.image_width)).transpose((1, 2, 0))

    def _reshape_optical_flow(self, optical_flow):
        return np.reshape(optical_flow, (2, self.image_height, self.image_width)).transpose((1, 2, 0))

    def step(self, action):
        if len(action.shape) == 2:
            action = np.reshape(action, (-1, 1))
        action = action.astype(np.float32)
        success = self.env.step(action)
        return success

    def render(self):
        success = self.env.render()
        return success

    def get_image(self):
        self.env.getImage(self.image)
        return self._reshape_image(self.image)

    def get_optical_flow(self):
        self.env.getOpticalFlow(self.optical_flow)
        return self._reshape_optical_flow(self.optical_flow)

    def get_state(self):
        self.env.getState(self.state)
        return self.state.copy()

    def get_sim_time_step(self):
        return self.env.getSimTimeStep()

    def set_sim_time_step(self, sim_time_step):
        self.env.setSimTimeStep(float(sim_time_step))

    def set_reduced_state(self, reduced_state):
        if len(reduced_state.shape) == 2:
            reduced_state = np.reshape(reduced_state, (-1, 1))
        reduced_state = reduced_state.astype(np.float32)
        self.env.setReducedState(reduced_state, reduced_state.shape[0])

    # def is_colliding(self):
    #     self.env.getCollision()

    def connect_unity(self, pub_port=10253, sub_port=10254):
        self.env.connectUnity(pub_port, sub_port)

    def disconnect_unity(self):
        self.env.disconnectUnity()


if __name__ == "__main__":
    def direct_encoding(flo, max_value=None, preserve_direction=False, **kwargs):
        if max_value is None:
            max_value = np.max(flo)

        if not preserve_direction:
            flo = np.clip(flo, -max_value, max_value)
            flo = flo / (max_value * 2)
        else:
            pass

        frame = (flo * 255.0 + 127.0).astype(np.uint8)
        frame = np.concatenate((frame, np.zeros(frame.shape[:2] + (1,), dtype=np.uint8)), axis=2)
        return frame


    import cv2
    from time import sleep

    current = np.array([-10.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    diff_straight = np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    diff_back = np.array([-0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    diff_left = np.array([0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
    diff_right = np.array([0.0, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
    diff_up = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0])
    diff_down = np.array([0.0, 0.0, -0.25, 0.0, 0.0, 0.0, 0.0])
    diff = diff_straight

    current = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    diff = np.array([0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0])

    env = RacingEnvWrapper(rendering_only=True)
    print(env.image_width, env.image_height)
    env.connect_unity()
    env.set_reduced_state(current)
    env.set_sim_time_step(0.1)

    c = 0
    while True:
        ret = env.render()
        if not ret:
            continue

        image = env.get_image()
        optical_flow = env.get_optical_flow()
        print("Optical flow:", optical_flow.min(), optical_flow.max(), np.sum(np.isnan(optical_flow)))
        optical_flow = direct_encoding(optical_flow)
        # print(optical_flow[:5, :5])
        # env.step(np.array([12.0, 0.0, 0.0, 0.0]))
        current += diff
        diff *= -1
        env.set_reduced_state(current)
        # sleep(0.1)
        # print("First RGB value in optical flow image (BGR):", optical_flow[0, 0])
        cv2.imshow("Image", image)
        cv2.imshow("Flow", optical_flow)
        k = cv2.waitKey(0) & 0xff
        if k == ord("q"):
            break

        c += 1
        continue
        if c > 20:
            diff = diff_back
        if c > 40:
            diff = diff_left
        if c > 60:
            diff = diff_right
        if c > 80:
            diff = diff_up
        if c > 100:
            diff = diff_down

    env.disconnect_unity()
