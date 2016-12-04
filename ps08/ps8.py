"""Problem Set 8: Motion History Images."""

import numpy as np
import cv2

# I/O directories
input_dir = "input"
output_dir = "output"


class MotionHistoryBuilder(object):
    """Builds a motion history image (MHI) from sequential video frames."""

    def __init__(self, frame, **kwargs):
        """Initialize motion history builder object.

        Parameters
        ----------
            frame (numpy.array): color BGR uint8 image of initial video frame, values in [0, 255]
            kwargs: additional keyword arguments needed by builder, e.g. theta, tau
        """
        # TODO: Your code here - initialize variables, read keyword arguments (if any)
        self.mhi = np.zeros(frame.shape[:2], dtype=np.float_)  # e.g. motion history image
        self.theta = kwargs.get('theta', 10)
        self.tau = kwargs.get('tau', 30)
        self.prev_frame = None

    def process(self, frame):
        """Process a frame of video, return binary image indicating motion areas.

        Parameters
        ----------
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255]

        Returns
        -------
            motion_image (numpy.array): binary image (type: bool or uint8), values: 0 (static) or 1 (moving)
        """
        # TODO: Your code here - compute binary motion image, update MHI
        blurred_frame = cv2.GaussianBlur(frame, (15, 15), 11)

        # if the first frame, return zeros motion_image
        if self.prev_frame is None:
            # update the prev_frame
            self.prev_frame = blurred_frame.copy()
            return np.zeros(frame.shape[:2], dtype=np.float_)

        frame_diff = np.abs(blurred_frame.astype(np.float) - self.prev_frame.astype(np.float))
        # B_t(x, y) = 1 if frame - prev_frame >= theta, 0 otherwise
        # since we want to threshold greater or equal (>=), we subtract theta with 1e-5
        _, motion_image = cv2.threshold(frame_diff.astype(np.float32), self.theta - 1e-5, 1., cv2.THRESH_BINARY)
        # without using grayscale, we need to average the motion to convert from 3d to 2d
        _, motion_image = cv2.threshold(np.mean(motion_image, axis=2).astype(np.float32), 0, 1., cv2.THRESH_BINARY)

        # update the prev_frame
        self.prev_frame = blurred_frame.copy()

        # # update the MHI
        self.mhi -= 1
        # if not moving, range between 0 and I(x,y,t) - 1
        _, self.mhi = cv2.threshold(self.mhi.astype(np.float32), 0., np.max(self.mhi), cv2.THRESH_TOZERO)
        # if moving
        self.mhi[np.where(motion_image >= 1.)[:2]] = self.tau

        # pass  # return motion_image  # note: make sure you return a binary image with 0s and 1s
        return motion_image

    def get_MHI(self):
        """Return motion history image computed so far.

        Returns
        -------
            mhi (numpy.array): float motion history image, values in [0.0, 1.0]
        """
        # TODO: Make sure MHI is updated in process(), perform any final steps here (e.g. normalize to [0, 1])
        # Note: This method may not be called for every frame (typically, only once)
        # normalize to [0, 1]
        return cv2.normalize(self.mhi, 0., 1., norm_type=cv2.NORM_MINMAX)


class Moments(object):
    """Spatial moments of an image - unscaled and scaled."""

    def __init__(self, image):
        """Compute spatial moments on given image.

        Parameters
        ----------
            image (numpy.array): single-channel image, uint8 or float
        """
        # TODO: Your code here - compute all desired moments here (recommended)
        self.central_moments = np.zeros((1, 8))  # array: [mu20, mu11, mu02, mu30, mu21, mu12, mu03, mu22]
        self.scaled_moments = np.zeros((1, 8))  # array: [nu20, nu11, nu02, nu30, nu21, nu12, nu03, nu22]
        # Note: Make sure computed moments are in correct order

    def get_central_moments(self):
        """Return central moments as NumPy array.

        Order: [mu20, mu11, mu02, mu30, mu21, mu12, mu03, mu22]

        Returns
        -------
            self.central_moments (numpy.array): float array of central moments
        """
        return self.central_moments  # note: make sure moments are in correct order

    def get_scaled_moments(self):
        """Return scaled central moments as NumPy array.

        Order: [nu20, nu11, nu02, nu30, nu21, nu12, nu03, nu22]

        Returns
        -------
            self.scaled_moments (numpy.array): float array of scaled central moments
        """
        return self.scaled_moments  # note: make sure moments are in correct order


def compute_feature_difference(a_features, b_features):
    """Compute feature difference between two videos.

    Parameters
    ----------
        a_features: features from one video, MHI & MEI moments in a 16-element 1D array
        b_features: like a_features, from other video

    Returns
    -------
        diff: a single float value, difference between the two feature vectors
    """
    # TODO: Your code here - return feature difference using an appropriate measure
    # Tip: Scale/weight difference values to get better results as moment magnitudes differ
    pass  # change to return diff
