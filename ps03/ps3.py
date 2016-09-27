import cv2
import numpy as np


def disparity_ssd(L, R, direction, w_size, dmax):
    """ Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    Using the Sum of Squared Differences
    Refer to https://software.intel.com/en-us/node/504333

    Parameters
    ----------
        L (numpy.array): Grayscale left image, in range [0.0, 1.0]
        R (numpy.array): Grayscale right image, same size as L
        direction (int):  if 1: the range of d should be [0, 1, 2, ..., dmax]
                          if 0: the range of d should be [0, -1, -2, ..., -dmax]
        w_size (int): window size, type int representing width and height (n, n)
        dmax (int): Maximum value of pixel disparity to test

    Returns
    -------
        disp_img (numpy.array): Disparity map of dtype = float, same size as L or R.
                                Return it without normalizing or clipping it.
    """

    # TODO: Your code here
    pass  # TODO: Change to return disp_img


def disparity_ncorr(L, R, direction, w_size, dmax):
    """ Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    Using the normalized correlation method


    Parameters
    ----------
        L (numpy.array): Grayscale left image, in range [0.0, 1.0]
        R (numpy.array): Grayscale right image, same size as L
        direction (int):  if 1: the range of d should be [0, dmax]
                          if 0: the range of d should be [-dmax, 0]
        w_size (int): window size, type int representing width and height (n, n)
        dmax (int): Maximum value of pixel disparity to test

    Returns
    -------
        disp_img (numpy.array): Disparity map of dtype = float, same size as L or R.
                                Return it without normalizing or clipping it.
    """

    # TODO: Your code here
    pass  # TODO: Change to return disp_img


def add_noise(image, sigma):
    """ Add Gaussian noise to the input image.
        See np.random.normal()

    Parameters
    ----------
        image (numpy.array): Input image, Numpy array i.e. L or R.
                             This array can either be of dtype = [int, float].
        sigma (float): Standard deviation of the distribution

    Returns
    -------
        noisy (numpy.array): Raw output image with added noise of dtype = float.
                             Return it without normalizing or clipping it.
    """

    # TODO: Your code here
    pass  # TODO: Change to return noisy


def increase_contrast(image, percent):
    """ Increase the input image contrast.

    Parameters
    ----------
        image (numpy.array): Input image, Numpy array i.e. L or R.
                             This array can either be of dtype = [int, float].
        percent (float): value to increase contrast. The autograder
                         uses percentage values (percentage i.e. 10 %).

    Returns
    -------
        img_out (numpy.array): Raw output image of dtype = float.
                               Return it without normalizing or clipping it.
    """

    # TODO: Your code here
    pass  # TODO: Change to return img_out
