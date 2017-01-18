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
    y, x = L.shape
    disparity_map = np.zeros((y, x, dmax))
    kernel = np.ones((w_size, w_size)) / (w_size * w_size)

    for d in xrange(0, dmax):
        if d == 0:
            rshift = R
        else:
            if direction == 0:
                rshift = cv2.copyMakeBorder(R[:, d:], 0, 0, 0, d, cv2.BORDER_REPLICATE)  # shift to left
            else:
                rshift = cv2.copyMakeBorder(R[:, :-d], 0, 0, d, 0, cv2.BORDER_REPLICATE)  # shift to right
        # https://software.intel.com/en-us/node/504333
        disparity_map[:, :, d] = cv2.filter2D((L - rshift) ** 2, -1, kernel)

    # get the minimum squared error
    disp_img = np.argmin(disparity_map, axis=2)

    return disp_img


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
    y, x = L.shape
    disparity_map = np.zeros((y, x, dmax))
    kernel = np.ones((w_size, w_size)) / (w_size * w_size)

    for d in xrange(0, dmax):
        if d == 0:
            rshift = R
        else:
            if direction == 0:
                rshift = cv2.copyMakeBorder(R[:, d:], 0, 0, 0, d, cv2.BORDER_REPLICATE)  # shift to left
            else:
                rshift = cv2.copyMakeBorder(R[:, :-d], 0, 0, d, 0, cv2.BORDER_REPLICATE)  # shift to right
        # https://software.intel.com/en-us/node/504333
        r_tx = cv2.filter2D(L * rshift, -1, kernel)
        r_xx = cv2.filter2D(rshift * rshift, -1, kernel)
        r_tt = cv2.filter2D(L * L, -1, kernel)
        disparity_map[:, :, d] = r_tx / np.sqrt(r_xx * r_tt)

    disp_img = np.argmax(disparity_map, axis=2)

    return disp_img


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
    noise = np.random.randn(*image.shape) * sigma
    output = image.astype(np.float) + noise
    return output


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
    return image * (1 + (percent / 100.))
