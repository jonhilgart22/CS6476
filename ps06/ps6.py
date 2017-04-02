"""Problem Set 6: Optic Flow."""

import numpy as np
import cv2


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image (uint8).
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0], beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you in this function. Additionally you should set cv2.Sobel's 'scale' parameter to
    one eighth.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output from cv2.Sobel.
    """
    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=1. / 8)


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you in this function. Additionally you should set cv2.Sobel's 'scale' parameter to
    one eighth.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction. Output from cv2.Sobel.
    """
    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=1./8)


def gaussianKernel(k_size, sigma):
    k = cv2.getGaussianKernel(k_size, sigma)
    kernel = np.outer(k, k)
    return kernel


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method similar to the approach used
    in the last problem sets.

    Note: Implement this method using the instructions in the lectures and the documentation.

    You are not allowed to use any OpenCV functions that are related to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted averages. Here we assume the kernel window is a
                      square so you will use the same value for both width and height.
        k_type (str): type of kernel to use for weighted averaging, 'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2. To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use 'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default value set to 1 because the autograder does not use
                       this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis, same size as the input images, floating-point
                             type.
            V (numpy.array): raw displacement (in pixels) along Y-axis, same size and type as U.
    """

    if k_type == 'uniform':
        # Generate a uniform kernel. The autograder uses this flag.
        kernel = np.ones((k_size, k_size)) / (k_size ** 2)
    elif k_type == 'gaussian':
        # Generate a gaussian kernel. This flag is not tested but may yield better results in some images.
        kernel = gaussianKernel(k_size, sigma)
    else:
        return None

    # Place your LK code here.
    Ix = gradient_x(img_a)
    Iy = gradient_y(img_a)
    It = img_b - img_a

    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    Ixt = Ix * It
    Iyt = Iy * It

    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)
    Sxt = cv2.filter2D(Ixt, -1, kernel)
    Syt = cv2.filter2D(Iyt, -1, kernel)

    U = (Sxy * Syt - Syy * Sxt) / (Sxx * Syy - Sxy * Sxy)
    V = (Sxy * Sxt - Sxx * Syt) / (Sxx * Syy - Sxy * Sxy)

    U[np.where(U == np.nan)] = 0.
    V[np.where(V == np.nan)] = 0.

    return U, V


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. When dealing with odd dimensions, the output image
    should be the result of rounding up the division by 2. For example (width, height): (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the input image.
    """
    # 5-tap filter
    alpha = 3./8  # same as np.array([1., 4., 6., 4., 1.]) / 16.
    kernel = np.array([1. - (alpha * 2.), 1., (alpha * 4.), 1., 1. - (alpha * 2.)]) / 4.

    # sub-sample every other row/col
    reduced_image = cv2.sepFilter2D(image, -1, kernel, kernel)[::2, ::2]
    return reduced_image


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is stored in a list of length equal the number of levels.
    The first element in the list ([0]) should contain the input image. All other levels contain a reduced version
    of the previous level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    g_pyr = [image]
    for _ in xrange(levels - 1):
        g_pyr.append(reduce_image(g_pyr[-1]))

    return g_pyr


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side, large to small from left to right.

    See the problem set instructions 2a. for a reference on how the output should look like.

    Make sure you call normalize_and_scale() for each image in the pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked from left to right.
    """
    img_shape = img_list[0].shape
    img_out = normalize_and_scale(img_list[0], scale_range=(0, 255))
    for img in img_list[1:]:
        out = np.zeros((img_shape[0], img.shape[1]))
        out[0:img.shape[0], 0:img.shape[1]] = normalize_and_scale(img, scale_range=(0, 255))[...]
        img_out = np.concatenate((img_out, out), axis=1)

    return img_out


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and width.
    """
    # 5-tap filter
    alpha = 3./8  # same as np.array([1., 4., 6., 4., 1.]) / 16.
    kernel = np.array([1. - (alpha * 2.), 1., (alpha * 4.), 1., 1. - (alpha * 2.)]) / 4.

    expanded_image = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    expanded_image[::2, ::2] = image[...]
    expanded_image = 4 * cv2.sepFilter2D(expanded_image, -1, kernel, kernel)

    return expanded_image


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    l_pyr = []
    for i in xrange(len(g_pyr) - 1):
        lapl = g_pyr[i] - expand_image(g_pyr[i + 1])[:g_pyr[i].shape[0], :g_pyr[i].shape[1]]
        l_pyr.append(lapl)

    l_pyr.append(g_pyr[-1])
    return l_pyr


def warp(image, U, V, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT101):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic interpolation and the BORDER_REFLECT101 border mode.
    You may change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in cv2.remap.

    Returns:
        numpy.array: warped image, such that warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    h, w = image.shape
    X, Y = np.meshgrid(xrange(w), xrange(h))

    # set type to np.float32 for cv2.remap
    X = (X + U).astype(np.float32)
    Y = (Y + V).astype(np.float32)

    warped = cv2.remap(image, X, Y, interpolation=interpolation, borderMode=border_mode)

    return warped


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT101):
    """Computes the optic flow using the Hierarchical Lucas-Kanade method.

    Refer to the problem set documentation to find out the steps involved in this function.

    This method should use reduce_image(), expand_image(), warp(), and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis, same size as the input images, floating-point
                             type.
            V (numpy.array): raw displacement (in pixels) along Y-axis, same size and type as U.
    """
    # create a list of reduced images to level k
    A = gaussian_pyramid(img_a, levels)
    B = gaussian_pyramid(img_b, levels)

    # initialize U and V to be zero images the size of smallest A
    U = np.zeros(A[-1].shape)
    V = np.zeros(A[-1].shape)

    # loop from the smallest A and B
    for A_k, B_k in reversed(zip(A, B)):
        kh, kw = A_k.shape

        # expand the flow field and double to get to the next level
        U = (expand_image(U) * 2)[:kh, :kw]
        V = (expand_image(V) * 2)[:kh, :kw]

        # warp B_k using U and V to form C_k
        C_k = warp(B_k, U, V, interpolation=interpolation, border_mode=border_mode)

        # perform LK on A_k and C_k to yield two incremental flow fields D_x and D_y
        D_x, D_y = optic_flow_lk(A_k, C_k, k_size, k_type, sigma)

        # add to the original flow
        U = U + D_x
        V = V + D_y

    return U, V
