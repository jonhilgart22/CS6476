"""Problem Set 6: Optic Flow."""

import numpy as np
import cv2


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Utility function that normalizes and scales an image to [0, 255]

    Parameters
    ----------
        image_in (numpy.array): Image in
        scale_range (tuple): Range values (min, max)

    Returns
    -------
        image_out (numpy.array): Image out (uint8)
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0], beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Compute image gradient in X direction. Use cv2.Sobel.

    Parameters
    ----------
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].

    Returns
    -------
        Ix (numpy.array): image gradient in X direction. Output from cv2.Sobel.
    """

    # TODO: Your code here
    pass


def gradient_y(image):
    """Compute image gradient in Y direction. Use cv2.Sobel.

    Parameters
    ----------
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].

    Returns
    -------
        Iy (numpy.array): image gradient in Y direction. Output from cv2.Sobel.
    """

    # TODO: Your code here
    pass


def optic_flow_LK(img_a, img_b, k_size, k_type, sigma=1):
    """Compute optic flow using the Lucas-Kanade method from the lectures.
    You are not allowed to use any OpenCV functions that are related to Optic Flow.

    Parameters
    ----------
        img_a (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted averages. Here we assume
                      the kernel window is a square so you will use the same value for both
                      width and height.
        k_type (str): type of filter to use for weighted averaging, 'uniform' or 'gaussian'.
                      By uniform we mean a kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use cv2.getGaussianKernel.
                      The autograder will use 'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default value set to 1 because the
                       autograder does not use this parameter.

    Returns
    -------
        U (numpy.array): raw displacement (in pixels) along X-axis, same size as image,
                         floating-point type
        V (numpy.array): raw displacement (in pixels) along Y-axis, same size and type as U
    """

    # TODO: Your code here
    if k_type == 'uniform':
        # Generate a uniform kernel. The autograder uses this flag.
        pass
    elif k_type == 'gaussian':
        # Generate a gaussian kernel. This flag is not tested but may yield
        # better results in some images.
        pass
    pass


def reduce(image):
    """Reduce image to the next smaller level. Follow the process shown in:
    The lecture 6B-L3. Look for the 5-tap separable filter. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid as a Compact Image Code
    You can find the link in the problem set instructions.

    Parameters
    ----------
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        reduced_image (numpy.array): same type as image, half size (rows*0.5, cols*0.5)
    """

    # TODO: Your code here
    pass


def gaussian_pyramid(image, levels):
    """Create a Gaussian pyramid of given image.
    Here you should call your reduce() function.

    Parameters
    ----------
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0]
        levels (int): number of levels in the resulting pyramid

    Returns
    -------
        g_pyr (list): Gaussian pyramid, list of numpy.arrays with g_pyr[0] = image
    """

    # TODO: Your code here
    pass


def create_combined_img(img_list):
    """Stack images from the input pyramid list side-by-side, large to small from left to right.
    See the problem set instructions 2a. for a reference on how the output should look like.
    Make sure you call normalize_and_scale() for each image in the pyramid when populating img_out.

    Parameters
    ----------
        img_list (list): List with pyramid images

    Returns
    -------
        img_out (numpy.array): Non-normalized output image
    """

    # TODO: Your code here
    pass


def expand(image):
    """Expand image to the next larger level. Follow the process shown in:
    The lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid as a Compact Image Code
    You can find the link in the problem set instructions.

    Parameters
    ----------
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        expanded_image (numpy.array): same type as image, double the size of the input image.
    """

    # TODO: Your code here
    pass


def laplacian_pyramid(g_pyr):
    """Create a Laplacian pyramid from a given Gaussian pyramid.
    Here you should call your expand() function.

    Parameters
    ----------
        g_pyr (list): Gaussian pyramid, as returned by gaussian_pyramid()

    Returns
    -------
        l_pyr (list): Laplacian pyramid, with l_pyr[-1] = g_pyr[-1]
    """

    # TODO: Your code here
    pass


def warp(image, U, V):
    """Warp image using X and Y displacements (U and V). Here you should use
    cv2.remap. To pass the autograder make sure you set interpolation to Cubic
    and the border mode BORDER_REFLECT101. You may change this to work with the
    problem set images.

    Parameters
    ----------
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        warped (numpy.array): warped image, such that
                              warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    # TODO: Your code here
    pass


def hierarchical_LK(img_a, img_b, levels, k_size, k_type, sigma):
    """Compute optic flow using the Hierarchical Lucas-Kanade method.
    You are not allowed to use any OpenCV functions that are related to Optic Flow.
    Here you should be using reduce(), expand(), warp(), and optic_flow_LK().

    Parameters
    ----------
        img_a (numpy.array): grayscale floating-point image, values in [0.0, 1.0]
        img_b (numpy.array): grayscale floating-point image, values in [0.0, 1.0]
        levels (int): Number of levels
        k_size (int): parameter to be passed to optic_flow_LK
        k_type (str): parameter to be passed to optic_flow_LK
        sigma (float): parameter to be passed to optic_flow_LK

    Returns
    -------
        U (numpy.array): raw displacement (in pixels) along X-axis, same size as image,
                         floating-point type
        V (numpy.array): raw displacement (in pixels) along Y-axis, same size and type as U
    """

    # TODO: Your code here
    pass