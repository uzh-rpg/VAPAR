import numpy as np
import pandas as pd

from collections import deque
from scipy.spatial.transform import Rotation
from skspatial.objects import Vector, Points, Line, Point, Plane
from shapely.geometry import LineString


########################
########################


class Checkpoint:
    """
    Checkpoint object represented as 2D surface in 3D space with x-axis
    pointing in the direction of flight/desired passing direction.

    Contains useful methods for determining distance of points,
    and intersections with the plane.
    """

    def __init__(self, df, dims=None, dtype='liftoff'):
        # set variable names according to the type of data
        if dtype == 'liftoff':
            position_varnames = ['px', 'py', 'pz']
            rotation_varnames = ['qx', 'qy', 'qz', 'qw']
            dimension_varnames = ['dy', 'dz']
            dimension_scaling_factor = 1.
        else:  # gazesim: default
            position_varnames = ['pos_x', 'pos_y', 'pos_z']
            rotation_varnames = ['rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat']
            dimension_varnames = ['dim_y', 'dim_z']
            dimension_scaling_factor = 2.5

        # current gate center position
        p = df[position_varnames].values.flatten()

        # current gate rotation quaternion
        q = df[rotation_varnames].values.flatten()

        # if no width and height dimensions were specified
        if dims is None:
            dims = (df[dimension_varnames[0]] * dimension_scaling_factor,
                    df[dimension_varnames[1]] * dimension_scaling_factor)

        # half width and height of the gate
        hw = dims[0] / 2.  # half width
        hh = dims[1] / 2.  # half height

        # assuming gates are oriented in the direction of flight: x=forward, y=left, z=up
        proto = np.array([[0.,  0.,  0.,  0.,  0.],  # assume surface, thus no thickness along x axis
                          [hw, -hw, -hw,  hw,  hw],
                          [hh,  hh, -hh, -hh,  hh]])
        self._corners = (Rotation.from_quat(q).apply(proto.T).T + p.reshape(3, 1)).astype(float)
        self._center = p
        self._rotation = q
        self._normal = Rotation.from_quat(q).apply(np.array([1, 0, 0]))
        self._width = dims[0]
        self._height = dims[1]

        # 1D minmax values
        self._x = np.array([np.min(self._corners[0, :]), np.max(self._corners[0, :])])
        self._y = np.array([np.min(self._corners[1, :]), np.max(self._corners[1, :])])
        self._z = np.array([np.min(self._corners[2, :]), np.max(self._corners[2, :])])

        # 2D line representations of gate horizontal axis
        self._xy = LineString([((np.min(self._corners[0, :])), np.min(self._corners[1, :])),
                               ((np.max(self._corners[0, :])), np.max(self._corners[1, :]))])
        self._xz = LineString([((np.min(self._corners[0, :])), np.min(self._corners[2, :])),
                               ((np.max(self._corners[0, :])), np.max(self._corners[2, :]))])
        self._yz = LineString([((np.min(self._corners[1, :])), np.min(self._corners[2, :])),
                               ((np.max(self._corners[1, :])), np.max(self._corners[2, :]))])

        # plane representation
        center_point = Point(list(self._center))
        normal_point = Point(list(self._center + self._normal))
        normal_vector = Vector.from_points(center_point, normal_point)
        self.plane = Plane(point=center_point, normal=normal_vector)

        self.x_axis = Line(point=self._corners[:, 0], direction=self._corners[:, 1] - self._corners[:, 0])
        self.y_axis = Line(point=self._corners[:, 0], direction=self._corners[:, 3] - self._corners[:, 0])
        self.length_x_axis = self.x_axis.direction.norm()
        self.length_y_axis = self.y_axis.direction.norm()

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def corners(self):
        return self._corners

    @property
    def center(self):
        return self._center

    @property
    def rotation(self):
        return self._rotation

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def xy(self):
        return self._xy

    @property
    def xz(self):
        return self._xz

    @property
    def yz(self):
        return self._yz

    def get_distance(self, p):
        return self.plane.distance_point(p)

    def get_signed_distance(self, p):
        return self.plane.distance_point_signed(p)

    def intersect(self, p0: np.ndarray([]), p1: np.ndarray([])) -> tuple():
        """
        Returns 2D and 3D intersection points with the gate surface and a
        line from two given points (p0, p1).
        """
        # Initialize relevant variables
        point_2d = None
        point_3d = None
        _x = None
        _y = None
        _z = None
        count = 0

        # only proceed if no nan values
        if (np.sum(np.isnan(p0).astype(int))==0) & (np.sum(np.isnan(p1).astype(int))==0):
            # line between start end end points
            line_xy = LineString([(p0[0], p0[1]),
                                  (p1[0], p1[1])])
            line_xz = LineString([(p0[0], p0[2]),
                                  (p1[0], p1[2])])
            line_yz = LineString([(p0[1], p0[2]),
                                  (p1[1], p1[2])])

            if self.xy.intersects(line_xy):
                count += 1
                _x, _y = [val for val in self.xy.intersection(line_xy).coords][0]

            if self.xz.intersects(line_xz):
                count += 1
                _x, _z = [val for val in self.xz.intersection(line_xz).coords][0]

            if self.yz.intersects(line_yz):
                count += 1
                _y, _z = [val for val in self.yz.intersection(line_yz).coords][0]

            # at least two of the three orthogonal lines need to be crossed
            if count > 1:
                point_3d = np.array([_x, _y, _z])
                point_2d = self.point2d(point_3d)

        return point_2d, point_3d

    def point2d(self, p: np.ndarray([])) -> np.ndarray([]):
        """
        Returns normalized [0-1 range] 2D coordinates of the intersection
        point within gate surface.

        The origin is the upper left corner (1st corner point)
        X-axis is to the right
        Y-axis is down
        """
        # Project the 3D intersection point onto the x and y surface axies.
        px_projected = self.x_axis.project_point(p)
        py_projected = self.y_axis.project_point(p)
        length_px_projected = self.x_axis.point.distance_point(px_projected)
        length_py_projected = self.y_axis.point.distance_point(py_projected)

        # Return the normalized 2D projection of the intersection point.
        return np.array([length_px_projected / self.length_x_axis,
                         length_py_projected / self.length_y_axis])


def detect_checkpoint_pass(
        t: np.array,
        px: np.array,
        py: np.array,
        pz: np.array,
        checkpoint: Checkpoint,
        distance_threshold: int=None
        ) -> np.array:
    """
    Return timestamps when drone passes a checkpoint from given drone
    timestamps (t), position (px, py, pz), and checkpoint object.

        t: timestamps in seconds,
        px, py, pz: drone position in x, y, z in meters
        checkpoint: Gate object, i.e. 2D surface in 3D space
        distance_threshold: distance threshold in meters for which to
            consider candidate sampling point for detecting gate interaction

        Update on 12.02.2021
        Checks if position data is within a distance threshold from the gate
        And for those data checks if the gate was passed
    """
    position = np.hstack((px.reshape(-1, 1), np.hstack((py.reshape(-1, 1), pz.reshape(-1, 1)))))

    # Set distance threshold to 60% of the gate surface diagonal.
    if distance_threshold is None:
        distance_threshold = 0.6 * np.sqrt(checkpoint.width ** 2 + checkpoint.height ** 2)

    # Select candidate timestamps in three steps:
    # First, find all timestamps when quad is close to gate.
    gate_center = checkpoint.center.reshape((1, 3)).astype(float)
    distance_from_gate_center = np.linalg.norm(position - gate_center, axis=1)
    timestamps_near_gate = t[distance_from_gate_center < distance_threshold]

    # Second, cluster the timestamps that occur consecutively
    dt = np.nanmedian(np.diff(t))
    ind = np.diff(timestamps_near_gate) > (4*dt)
    if len(ind) == 0:
        return []
    ind1 = np.hstack((ind, True))
    ind0 = np.hstack((True, ind))
    timestamp_clusters = np.hstack((
        timestamps_near_gate[ind0].reshape(-1, 1),
        timestamps_near_gate[ind1].reshape(-1, 1)
    ))

    # Third, find gate passing events using signed distances from gate plane.
    event_timestamps = []
    for cluster in range(timestamp_clusters.shape[0]):
        start_time = timestamp_clusters[cluster, 0]
        end_time = timestamp_clusters[cluster, 1]
        ind = (t>=start_time) & (t<=end_time)
        curr_time = t[ind]
        curr_position = position[ind, :]
        curr_signed_distance = np.array([
            checkpoint.get_signed_distance(curr_position[i, :]) for i in range(
                curr_position.shape[0]
            )
        ])
        # Find transitions from negative to positive signed distance.
        #  where "negative distance" is behind the gate (negative x in gate
        #  frame) and "positive distance" is in front of the gate (positive x
        #  in gate frame.
        ind = ((curr_signed_distance <= 0)
               & (np.diff(np.hstack((curr_signed_distance, curr_signed_distance[-1])) > 0) == 1))
        if np.sum(ind) > 0:
            event_timestamps.append(curr_time[ind][0])

    return event_timestamps


########################
########################


class GateDirectionHighLevelLabel:

    def __init__(self, config):
        self.decision_threshold = config.gate_direction_branching_threshold
        self.return_extra_info = config.return_extra_info
        self.original_checkpoint_index = config.gate_direction_start_gate

        z_add = 3.0 if "wave" in config.trajectory_path else 0.0
        track = pd.DataFrame({
            "px": [-1.3, -18, -25, -18, -1.3, 1.3, 18, 25, 18, 1.3],
            "py": [1.3, 10, 0, -10, -1.3, 1.3, 10, 0, -10, -1.3],
            "pz": [2.1 + z_add, 2.1, 2.1 + z_add, 2.1, 2.1 + z_add, 2.1 + z_add, 2.1, 2.1 + z_add, 2.1, 2.1 + z_add],
            "qx": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "qy": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "qz": [-0.923879539192906, -0.923879539192906, 0.707106781186547, 0.382683416234233,
                   -0.382683416234233, -0.382683416234233, 0.382683416234233, 0.707106781186547,
                   -0.923879539192906, -0.923879539192906],
            "qw": [-0.382683416234233, 0.382683416234233, -0.707106781186548, -0.923879539192906,
                   -0.923879539192906, -0.923879539192906, -0.923879539192906, -0.707106781186548,
                   0.382683416234233, -0.382683416234233],
            "dx": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "dy": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "dz": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        })

        # track/gate definition TODO: need to be in correct order!! depends on starting position
        # => how do we figure this out? hard-coding for assumed MPC start would be kind of fine
        #    but I guess the least you could do is make it a "specifiable" parameter
        self.checkpoints = [Checkpoint(track.iloc[i]) for i in range(10)]
        self.current_checkpoint_index = config.gate_direction_start_gate
        self.current_checkpoint = self.checkpoints[self.current_checkpoint_index]

        # keep track of history of positions to detect gate passing event
        self.history_length = 200
        self.history_time = deque(maxlen=self.history_length)
        self.history_pos_x = deque(maxlen=self.history_length)
        self.history_pos_y = deque(maxlen=self.history_length)
        self.history_pos_z = deque(maxlen=self.history_length)

    def reset(self):
        self.current_checkpoint_index = self.original_checkpoint_index
        self.current_checkpoint = self.checkpoints[self.current_checkpoint_index]

        self.history_time.clear()
        self.history_time.clear()
        self.history_pos_x.clear()
        self.history_pos_y.clear()
        self.history_pos_z.clear()

    def get_label(self, drone_state, simulation_time):
        extra_info = {}

        # drone_state = kwargs.get("drone_state", np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        drone_pos = drone_state[:3]
        drone_vel = drone_state[7:10]

        self.history_time.append(simulation_time)
        self.history_pos_x.append(drone_pos[0])
        self.history_pos_y.append(drone_pos[1])
        self.history_pos_z.append(drone_pos[2])

        # check whether we passed the "current" next gate
        passing_events = detect_checkpoint_pass(
            np.array(self.history_time),
            np.array(self.history_pos_x),
            np.array(self.history_pos_y),
            np.array(self.history_pos_z),
            self.current_checkpoint,
        )
        if len(passing_events) > 0:
            self.current_checkpoint_index = (self.current_checkpoint_index + 1) % len(self.checkpoints)
            self.current_checkpoint = self.checkpoints[self.current_checkpoint_index]
        extra_info["next_gate"] = self.current_checkpoint_index

        # define the direction vector
        direction_3d = self.current_checkpoint.center - drone_pos
        direction_3d = direction_3d / np.linalg.norm(direction_3d)
        extra_info["direction_3d_x"] = direction_3d[0]
        extra_info["direction_3d_y"] = direction_3d[1]
        extra_info["direction_3d_z"] = direction_3d[2]

        # print("checkpoint center: {}, drone position: {}, difference vector: {}, drone vel: {}".format(
        #     self.current_checkpoint.center, drone_pos, direction_3d, drone_vel))

        # compute the signed angle between the direction-to-next-gate vector and the velocity vector
        drone_vel_2d = drone_vel[[0, 1]]
        drone_vel_2d = drone_vel_2d / np.linalg.norm(drone_vel_2d)
        direction_2d = direction_3d[[0, 1]]
        direction_2d = direction_2d / np.linalg.norm(direction_2d)
        angle = np.arctan2(drone_vel_2d[1], drone_vel_2d[0]) - np.arctan2(direction_2d[1], direction_2d[0])
        # print("Angle (1): {} (rad), {} (deg)".format(angle, angle * 180.0 / np.pi))
        if np.abs(angle) > np.pi:
            if angle < 0.0:
                # angle = 2 * np.pi - np.abs(angle)
                angle = angle + 2 * np.pi
                # print("Angle (2): {} (rad), {} (deg)".format(angle, angle * 180.0 / np.pi))
            else:
                # angle = -(2 * np.pi - np.abs(angle))
                angle = angle - 2 * np.pi
                # print("Angle (3): {} (rad), {} (deg)".format(angle, angle * 180.0 / np.pi))

            """
            if np.abs(angle) > 0.0:
                angle = -angle
                print("Angle (4): {} (rad), {} (deg)".format(angle, angle * 180.0 / np.pi))
            """

        # print("Angle: {} (rad), {} (deg)".format(angle, angle * 180.0 / np.pi))
        extra_info["angle_rad"] = angle
        angle = angle * 180.0 / np.pi
        extra_info["angle_deg"] = angle

        high_level_label = 0
        if angle > self.decision_threshold:
            high_level_label = 1
        elif angle < -self.decision_threshold:
            high_level_label = 2

        extra_info["high_level_label"] = high_level_label

        # print("High-level label:", high_level_label, "\n")

        if self.return_extra_info:
            return high_level_label, extra_info
        return high_level_label
