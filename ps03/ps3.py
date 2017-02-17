import cv2
import numpy as np


def disparity_ssd(img1, img2, direction, w_size, dmax):
    """Returns a disparity map D(y, x) using the Sum of Squared Differences.

    Assuming img1 and img2 are the left (L) and right (R) images from the same scene. The disparity image contains
    values such that: L(y, x) = R(y, x) + D(y, x) when matching from left (L) to right (R).

    This method uses the Sum of Squared Differences as an error metric. Refer to:
    https://software.intel.com/en-us/node/504333

    The algorithm used in this method follows the pseudocode:

    height: number of rows in img1 or img2.
    width: number of columns in img1 or img2.
    DSI: initial array containing only zeros of shape (height, width, dmax)
    kernel: array of shape (w_size[0], w_size[1]) where each value equals to 1/(w_size[0] * w_size[1]). This allows
            a uniform distribution that sums to 1.

    for d going from 0 to dmax:
        shift = some_image_shift_function(img2, d)
        diff = img1 - shift  # SSD
        Square every element values  # SSD
        Run a 2D correlation filter (i.e. cv.filter2D) using the kernel defined above
        Save the results in DSI(:, :, d)

    For each location r, c the SSD for an offset d is in DSI(r,c,d). The best match for pixel r,c is represented by
    the index d for which DSI(r,c,d) is smallest.

    Args:
        img1 (numpy.array): grayscale image, in range [0.0, 1.0].
        img2 (numpy.array): grayscale image, in range [0.0, 1.0] same shape as img1.
        direction (int): if 1: match right to left (shift img1 left).
                         if 0: match left to right (shift img2 right).
        w_size (tuple): window size, type int representing both height and width (h, w).
        dmax (int): maximum value of pixel disparity to test.

    Returns:
        numpy.array: Disparity map of type int64, 2-D array of the same shape as img1 or img2.
                     This array contains the d values representing how far a certain pixel has been displaced.
                     Return without normalizing or clipping.
    """
    y, x = img1.shape
    disparity_map = np.zeros((y, x, dmax))
    kernel = np.ones(w_size) / (w_size[0] * w_size[1])

    if direction == 0:
        for d in xrange(0, dmax):
            shift = cv2.copyMakeBorder(img2[:, :(-d or None)], 0, 0, d, 0, cv2.BORDER_REPLICATE)
            disparity_map[:, :, d] = cv2.filter2D((img1 - shift) ** 2, -1, kernel)
    elif direction == 1:
        for d in xrange(0, dmax):
            shift = cv2.copyMakeBorder(img1[:, d:], 0, 0, 0, d, cv2.BORDER_REPLICATE)
            disparity_map[:, :, d] = cv2.filter2D((img2 - shift) ** 2, -1, kernel)

    disp_img = np.argmin(disparity_map, axis=2)

    return disp_img


def disparity_ncorr(img1, img2, direction, w_size, dmax):
    """Returns a disparity map D(y, x) using the normalized correlation method.

    This method uses a similar approach used in disparity_ssd replacing SDD with the normalized correlation metric.

    For more information refer to:
    https://software.intel.com/en-us/node/504333

    Unlike SSD, the best match for pixel r,c is represented by the index d for which DSI(r,c,d) is highest.

    Args:
        img1 (numpy.array): grayscale image, in range [0.0, 1.0].
        img2 (numpy.array): grayscale image, in range [0.0, 1.0] same shape as img1.
        direction (int): if 1: match right to left (shift img1 left).
                         if 0: match left to right (shift img2 right).
        w_size (tuple): window size, type int representing both height and width (h, w).
        dmax (int): maximum value of pixel disparity to test.

    Returns:
        numpy.array: Disparity map of type int64, 2-D array of the same shape size as img1 or img2.
                     This array contains the d values representing how far a certain pixel has been displaced.
                     Return without normalizing or clipping.
    """
    y, x = img1.shape
    disparity_map = np.zeros((y, x, dmax))
    kernel = np.ones(w_size) / (w_size[0] * w_size[1])

    if direction == 0:
        for d in xrange(0, dmax):
            shift = cv2.copyMakeBorder(img2[:, :(-d or None)], 0, 0, d, 0, cv2.BORDER_REPLICATE)
            # https://software.intel.com/en-us/node/504333
            r_tx = cv2.filter2D(img1 * shift, -1, kernel)
            r_xx = cv2.filter2D(shift * shift, -1, kernel)
            r_tt = cv2.filter2D(img1 * img1, -1, kernel)
            disparity_map[:, :, d] = r_tx / np.sqrt(r_xx * r_tt)
    elif direction == 1:
        for d in xrange(0, dmax):
            shift = cv2.copyMakeBorder(img1[:, d:], 0, 0, 0, d, cv2.BORDER_REPLICATE)
            # https://software.intel.com/en-us/node/504333
            r_tx = cv2.filter2D(img2 * shift, -1, kernel)
            r_xx = cv2.filter2D(shift * shift, -1, kernel)
            r_tt = cv2.filter2D(img2 * img2, -1, kernel)
            disparity_map[:, :, d] = r_tx / np.sqrt(r_xx * r_tt)

    disp_img = np.argmax(disparity_map, axis=2)

    return disp_img


def add_noise(img, sigma):
    """Returns a copy of the input image with gaussian noise added. The Gaussian noise mean must be zero.
    The parameter sigma controls the standard deviation of the noise.

    Args:
        img (numpy.array): input image of type int or float.
        sigma (float): gaussian noise standard deviation.

    Returns:
        numpy.array: output image with added noise of type float64. Return it without normalizing or clipping it.
    """
    noise = np.random.randn(*img.shape) * sigma
    output = img.astype(np.float) + noise
    return output


def increase_contrast(img, percent):
    """Returns a copy of the input image with an added contrast by a percentage factor.

    Args:
        img (numpy.array): input image of type int or float.
        percent (int or float): value to increase contrast. The autograder uses percentage values i.e. 10%.

    Returns:
        numpy.array: output image with added noise of type float64. Return it without normalizing or clipping it.
    """
    return img * (1 + (percent / 100.))
