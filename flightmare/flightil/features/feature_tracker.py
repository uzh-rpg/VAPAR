import cv2
import numpy as np

from time import time


class FeatureTracker:

    def __init__(self, max_features_to_track=100, static_mask=None):
        self.max_features_to_track = max_features_to_track

        # Shi-Tomasi feature detection parameters
        self.extraction_quality_level = 0.3
        self.extraction_min_distance = 7
        self.extraction_block_size = 7
        self.extraction_params = {
            "maxCorners": self.max_features_to_track,
            "qualityLevel": self.extraction_quality_level,
            "minDistance": self.extraction_min_distance,
            "blockSize": self.extraction_block_size,
        }

        # Lukas-Kanade optical flow parameters
        self.tracking_window_size = (21, 21)
        self.tracking_max_level = 2
        self.tracking_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        self.tracking_params = {
            "winSize": self.tracking_window_size,
            "maxLevel": self.tracking_max_level,
            "criteria": self.tracking_criteria,
        }

        # the parameters above are mostly just taken from the OpenCV tutorial on optical flow

        # ID counter
        self.id_counter = 0
        self.previous_time = -1

        # images
        self.image_shape = None
        self.previous_image = None
        self.static_mask = static_mask

        # feature information: [id, tracking_count, pos_x, pos_y, vel_x, vel_y]
        self.ids = []
        self.tracking_counts = []
        self.previous_points = None
        self.previous_norm_points_dict = None

        # TODO: maybe have some structure surrounding this that takes care of ensuring that the image can be processed
        #  and/or that feature tracks are "published" at the appropriate rate (similar to the ROS node in the original)

        # self.K = np.array([[0.41 * 800.0, 0., 0.5 * 800.0], [0., 0.56 * 600.0, 0.5 * 600.0], [0., 0., 1.]])
        self.K = np.array([[400.0, 0., 400.0], [0., 300.0, 300.0], [0., 0., 1.]])
        # self.K = np.array([[0.41, 0., 0.5], [0., 0.56, 0.5], [0., 0., 1.]])
        self.K_inv = np.linalg.inv(self.K)
        self.F = 800

    def reset(self):
        self.ids = []
        self.tracking_counts = []
        self.image_shape = None
        self.previous_image = None
        self.previous_points = None
        self.previous_norm_points_dict = None
        self.previous_time = -1

    def _filter_by_status(self, status, *args):
        # TODO: maybe filter ids, track_counts, previous_points "automatically" and args in addition to that?
        updated = []
        for arg in args:
            if isinstance(arg, list):
                updated.append([arg[i] for i in range(len(status)) if status[i] == 1])
            elif isinstance(arg, np.ndarray):
                updated.append(arg[status == 1])
        return tuple(updated)

    def _in_border(self, points, status):
        for p_idx, p in enumerate(points):
            if not (0 <= p[0] < self.image_shape[1] and 0 <= p[1] < self.image_shape[0]):
                status[p_idx] = 0
        return status

    def _filter_outliers(self, current_points):
        _, status = cv2.findFundamentalMat(self.previous_points, current_points, cv2.FM_RANSAC, 1.0, 0.99, 2000)

        if status is None:
            return current_points[:0]

        status = status.reshape(-1)

        self.ids, self.tracking_counts, self.previous_points, current_points = self._filter_by_status(
            status, self.ids, self.tracking_counts, self.previous_points, current_points)

        # print((status == 0).sum())

        """
        # I think this would pretty much just do what I do manually below (i.e. multiple by K_inv)
        test = cv2.undistortPoints(self.previous_points, self.K, None)
        # print(self.previous_points[0, :], "-", test[0, :])

        temp_previous = np.hstack((self.previous_points, np.ones((self.previous_points.shape[0], 1),
                                                                 dtype=self.previous_points.dtype)))
        temp_previous = (self.K_inv @ temp_previous.T).T
        temp_previous = temp_previous[:, :2] / temp_previous[:, 2:]
        temp_previous = temp_previous * self.F + np.array([400.0, 300.0], dtype=self.previous_points.dtype)[None, :]

        temp_current = np.hstack((current_points, np.ones((current_points.shape[0], 1), dtype=current_points.dtype)))
        temp_current = (self.K_inv @ temp_current.T).T
        temp_current = temp_current[:, :2] / temp_current[:, 2:]
        temp_current = temp_current * self.F + np.array([400.0, 300.0], dtype=temp_current.dtype)[None, :]

        _, temp_status = cv2.findFundamentalMat(temp_previous, temp_current, cv2.FM_RANSAC, 1.0, 0.99, 2000)
        """

        # print((status == 0).sum(), "-", (temp_status == 0).sum(), "-", all(status == temp_status))
        # print(self.previous_points[0, :], "-", temp_previous[0, :])
        # print()

        return current_points

    def process_image(self, image, current_time=None, return_image_points=False):
        if self.image_shape is None:
            self.image_shape = image.shape[:2]

        assert image.dtype == np.uint8, "Image needs to have type UINT8."
        assert all(image.shape[i] == self.image_shape[i] for i in range(len(self.image_shape))), "Shape mismatch."

        image = image.squeeze()
        if len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # get the current time to compute the velocity
        # TODO: this needs to be simulation time and not real time!
        if current_time is None:
            current_time = time()

        # TODO: potentially apply CLAHE/equalisation

        # set the previous image if it is none
        first_iteration = False
        if self.previous_image is None:
            self.previous_image = image
            first_iteration = True

        # create mask for existing features
        feature_mask = np.full_like(image, 255)

        # compute the optical flow/track features between previous and current image if it is not the first iteration
        current_points = self.previous_points
        if not first_iteration:
            # calculate the optical flow
            current_points, status, error = cv2.calcOpticalFlowPyrLK(self.previous_image, image, self.previous_points,
                                                                     None, **self.tracking_params)
            current_points = current_points.reshape(-1, 2)
            status = status.reshape(-1)

            # make sure the matched points are within the image boundaries
            status = self._in_border(current_points, status)

            # select only the feature information for points that could be matched
            self.ids, self.tracking_counts, self.previous_points, current_points = self._filter_by_status(
                status, self.ids, self.tracking_counts, self.previous_points, current_points)

            self.tracking_counts = [tc + 1 for tc in self.tracking_counts]
            """
            self.ids = [self.ids[i] for i in range(len(status)) if status[i] == 1]
            self.tracking_counts = [self.tracking_counts[i] + 1 for i in range(len(status)) if status[i] == 1]
            self.previous_points = self.previous_points[status == 1]
            current_points = current_points[status == 1]
            """

            # update mask
            for p in current_points:
                feature_mask = cv2.circle(feature_mask, tuple(p), self.extraction_block_size, 0, -1)
            if self.static_mask is not None:
                feature_mask = cv2.bitwise_and(feature_mask, self.static_mask)

        matched_points = 0
        if current_points is not None:
            matched_points = len(current_points)

            # TODO: if there are any point matches, filter out outliers
            current_points = self._filter_outliers(current_points)

            if len(current_points) == 0:
                print("\n[FeatureTracker] No point matches, need to reset.\n")
                self.reset()
                # TODO: should probably just act like this is the first iteration though...
                #  => actually, instead of doing the stuff below, could just reset/empty all the arrays I think
                #     and then new features should be computed automatically...
                # self.previous_image = image
                # first_iteration = True
                # current_points = None
                # return None

        # extract features if there aren't enough being tracked already
        current_points_count = 0 if current_points is None else len(current_points)
        additional_points_count = self.max_features_to_track - current_points_count
        if additional_points_count > 0:
            # extract features
            additional_points = cv2.goodFeaturesToTrack(image, mask=feature_mask, **self.extraction_params)

            if additional_points is not None:
                additional_points = additional_points.reshape(-1, 2)
                additional_points_count = min(additional_points_count, len(additional_points))

                # add the ids of the features and "initialise" the tracking counts
                for f_idx in range(additional_points_count):
                    self.ids.append(self.id_counter)
                    self.tracking_counts.append(0)
                    self.id_counter += 1

                # either initialise the features/points to track or add the new ones to them
                if current_points is None:
                    current_points = additional_points
                else:
                    current_points = np.concatenate((current_points, additional_points[:additional_points_count]), axis=0)
            elif current_points_count == 0:
                print("\n[FeatureTracker] No features to track found, need to reset.\n")
                self.reset()
                return None

        # "un-distort" points (although we don't have any distortion, at least not in the original data)
        # => essentially just applies the inverse of the intrinsic matrix to the points, which should give them
        #    as normalised coordinates on the image plane (i.e. in the range [-1, 1])
        current_norm_points = cv2.undistortPoints(current_points, self.K, None)
        current_norm_points = current_norm_points.reshape(-1, 2)

        # velocity calculation
        if self.previous_norm_points_dict is not None:
            current_velocities = []
            time_diff = current_time - self.previous_time
            for i in range(len(self.ids)):
                if self.ids[i] in self.previous_norm_points_dict:
                    velocity = (current_norm_points[i, :] - self.previous_norm_points_dict[self.ids[i]]) / time_diff
                    current_velocities.append(velocity)
                else:
                    current_velocities.append((0.0, 0.0))
        else:
            current_velocities = [(0.0, 0.0) for _ in range(len(self.ids))]

        #
        return_previous_points = self.previous_points if self.previous_points is not None else np.array([])
        self.previous_points = current_points
        self.previous_norm_points_dict = {self.ids[i]: current_norm_points[i, :]
                                          for i in range(len(self.previous_points))}

        # update the image to keep for matching
        self.previous_image = image

        # update the time for velocity computation
        self.previous_time = current_time

        # construct and return feature tracks
        # TODO: maybe convert to numpy array?
        # TODO: the points should probably be in normalised image coordinates (seems like that's what's used in DDA)
        #  => need to do all computations with "normal" image coordinates, but return normalised ones
        #  => should also store the previous normalised/undistorted points for velocity calculation
        features = [[self.ids[i], self.tracking_counts[i],
                     current_norm_points[i, 0], current_norm_points[i, 1],
                     current_velocities[i][0], current_velocities[i][1]]
                    for i in range(len(self.ids))]
        features = np.array(features)

        # print(np.median([f[1] for f in features]))

        if return_image_points:
            return features, return_previous_points, current_points, matched_points
        return features
