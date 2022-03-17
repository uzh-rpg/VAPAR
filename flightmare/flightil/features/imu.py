import numpy as np

from collections import deque
from scipy.spatial.transform import Rotation


class IMU:

    def __init__(self, base_frequency=100):
        # parameters: noise magnitude, maybe init state?,
        self.est_pos = None  # position
        self.est_rot = None  # rotation as quaternion
        self.est_vel = None  # linear velocity
        self.est_omega = None  # body rates
        self.est_acc = None  # linear acceleration

        # figure out length of buffer to compute acceleration over ~200ms
        self.base_time_step = 1.0 / base_frequency
        self.buffer_length = int(0.2 / self.base_time_step)
        self.buffer = deque(maxlen=self.buffer_length)

    def reset(self, init_state):
        # maybe only position and orientation?
        self.est_pos = init_state[:3]
        self.est_rot = init_state[3:7]
        self.est_vel = init_state[7:10]
        if init_state.shape[0] > 10:
            self.est_omega = init_state[10:13]
        if init_state.shape[0] > 13:
            self.est_acc = init_state[13:16]

    def update(self, omega, acc, time_step):
        # add noise to omega and acc
        # integrate based on most recent measurement
        pass

    def get_state_estimate(self):
        return np.concatenate((self.est_pos, self.est_rot, self.est_vel, self.est_omega, self.est_acc))


class IMURawMeasurements:

    def __init__(self, base_frequency=100):
        # figure out length of buffer to compute acceleration over ~200ms
        self.base_time_step = 1.0 / base_frequency
        self.buffer_length = int(0.2 / self.base_time_step)

        # create the buffers/queues for continuously updating the relevant vars
        self.buffer_pos = deque(maxlen=self.buffer_length)
        self.buffer_rot = deque(maxlen=self.buffer_length)
        self.buffer_time = deque(maxlen=self.buffer_length)

    def reset(self):
        self.buffer_pos.clear()
        self.buffer_rot.clear()
        self.buffer_time.clear()

    def get_state_estimate(self, state, time):
        self.buffer_pos.append(state[:3])
        self.buffer_rot.append(state[3:7])
        self.buffer_time.append(time)

        # if queues aren"t full yet, return zeros?
        if len(self.buffer_pos) != self.buffer_length:
            return np.concatenate((state[10:13], np.zeros((3,))))

        # convert the buffers into numpy arrays
        pos = np.vstack(self.buffer_pos)
        rot = np.vstack(self.buffer_rot)
        time = np.vstack(self.buffer_time)

        # compute the acceleration
        acc_world_frame = self._position2acceleration(time, pos)
        acc_body_frame = self._acceleration_world2body(rot, acc_world_frame)
        acc_body_frame = acc_body_frame[-1, :]

        # return only angular velocity and linear acceleration, i.e. "raw measurements"
        return np.concatenate((state[10:13], acc_body_frame))

    def _smooth_signal(self, x: np.array, window_len: int = 11, window: str = "hanning") -> np.array:
        """smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from "flat", "hanning", "hamming", "bartlett", "blackman"
                flat window will produce a moving average smoothing.
        output:
            the smoothed signal
        example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise ValueError("Smooth only accepts 1 dimensional arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
            raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        # print(len(s))
        if window == "flat":  # moving average
            w = np.ones(window_len, "d")
        else:
            w = eval("np." + window + "(window_len)")

        y = np.convolve(w / w.sum(), s, mode="valid")
        return y

    def _position2velocity(self, t: np.array, p: np.array, smoothing_width: float = 0.2) -> np.array:
        """
        Velocity smoothed from position data.
        Assumes regular sampling.
        """
        if t.shape[0] < 2:
            return np.zeros(p.shape)
        sr = 1 / np.nanmedian(np.diff(t))

        # Compute velocity
        v = np.diff(p, axis=0) * sr
        v = np.vstack((v, v[-1:, :]))

        # Signal smoothing.
        window = "blackman"
        window_len = int(sr * smoothing_width)
        if window_len > t.shape[0]:
            window_len = t.shape[0]
        v_smooth = np.empty(v.shape)
        for i in range(v.shape[1]):
            v_smooth[:, i] = self._smooth_signal(
                v[:, i], window_len=window_len, window=window
            )[window_len // 2:v.shape[0] + window_len // 2]

        return v_smooth

    def _position2acceleration(self, t: np.array, p: np.array, smoothing_width: float = 0.2) -> np.array:
        """
        Acceleration smoothed from position data. Assumes regular sampling.
        """
        if t.shape[0] < 3:
            return np.zeros(p.shape)
        sr = 1 / np.nanmedian(np.diff(t))

        # Compute acceleration.
        a = np.diff(p, n=2, axis=0) * (sr ** 2)
        a = np.vstack((a, a[-1:, :]))
        a = np.vstack((a, a[-1:, :]))

        # Signal smoothing.
        window = "blackman"
        window_len = int(sr * smoothing_width)
        if window_len > t.shape[0]:
            window_len = t.shape[0]
        a_smooth = np.empty(a.shape)
        for i in range(a.shape[1]):
            a_smooth[:, i] = self._smooth_signal(
                a[:, i], window_len=window_len, window=window
            )[window_len // 2:a.shape[0] + window_len // 2]

        # Add gravity.
        a_smooth[:, 2] -= 9.80665
        return a_smooth

    def _acceleration_world2body(self, q: np.array, a: np.array) -> np.array:
        """
        Returns acceleration in body frame from given rotation quaternion and
        acceleration in world frame.
        """
        return Rotation.from_quat(q).inv().apply(a)
