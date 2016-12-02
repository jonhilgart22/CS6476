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
        pass  # return motion_image  # note: make sure you return a binary image with 0s and 1s

    def get_MHI(self):
        """Return motion history image computed so far.

        Returns
        -------
            mhi (numpy.array): float motion history image, values in [0.0, 1.0]
        """
        # TODO: Make sure MHI is updated in process(), perform any final steps here (e.g. normalize to [0, 1])
        # Note: This method may not be called for every frame (typically, only once)
        return self.mhi


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
