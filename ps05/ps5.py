"""Problem Set 5: Harris, ORB, RANSAC."""

import numpy as np
import cv2


def gradient_x(image):
    """Compute image gradient in X direction. See cv2.Sobel.

    Parameters
    ----------
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Ix (numpy.array): image gradient in X direction, values in [-1.0, 1.0]
    """

    # TODO: Your code here
    pass  # TODO: Change to return Ix


def gradient_y(image):
    """Compute image gradient in Y direction. See cv2.Sobel.

    Parameters
    ----------
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Iy (numpy.array): image gradient in Y direction, values in [-1.0, 1.0]
    """

    # TODO: Your code here
    pass  # TODO: Change to return Iy


def make_image_pair(image1, image2):
    """Adjoin two images side-by-side to make a single new image.
       The output dimensions must take the maximum height from both images for the total height.
       The total width is found by adding the width of image1 and image2.

    Parameters
    ----------
        image1 (numpy.array): first image, could be grayscale or color (BGR).
                              This would take the left side of the output image.
        image2 (numpy.array): second image, could be grayscale or color (BGR).
                              This would take the right side of the output image.

    Returns
    -------
        image_pair (numpy.array): combination of both images, side-by-side, same type.
    """

    # TODO: Your code here
    pass  # TODO: Change to return image_pair


def harris_response(Ix, Iy, kernel, alpha):
    """Compute Harris response map using given image gradients.

    Parameters
    ----------
        Ix (numpy.array): image gradient in X direction, values in [-1.0, 1.0]
        Iy (numpy.array): image gradient in Y direction, same size and type as Ix
        kernel (tuple): 2D windowing kernel dimensions. ie. (3, 3)  (3, 5)
        alpha (float): Harris detector parameter multiplied with the square of trace

    Returns
    -------
        R (numpy.array): Harris response map, same size as inputs, floating-point
    """

    # TODO: Your code here
    pass  # TODO: Change to return R


def find_corners(R, threshold, radius):
    """Find corners in a given response map. Make sure you account for duplicate and overlapping points.

    Parameters
    ----------
        R (numpy.array): floating-point response map, e.g. output from the Harris detector
        threshold (float): Value between 0.0 and 1.0. Response values less than this should
                           not be considered plausible corners.
        radius (int): radius of circular region for non-maximal suppression
                      (could be half the side of square instead)

    Returns
    -------
        corners (numpy.array): peaks found in response map R, each row must be defined as [x, y].
                               Array size must be N x 2, where N are the number of points found.
    """

    # Normalize R
    R_norm = cv2.normalize(R, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # # Do not modify the code above. Continue working with R_norm. # #

    # TODO: Your code here
    pass  # TODO: Change to return corners


def draw_corners(image, corners):
    """Draw corners on (a copy of) given image.

    Parameters
    ----------
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0]
        corners (numpy.array): peaks found in response map R, as a sequence of [x, y] coordinates.
                               Array size must be N x 2, where N are the number of points found.

    Returns
    -------
        image_out (numpy.array): copy of image with corners drawn on it, in color (BGR).
    """

    # TODO: Your code here
    pass  # TODO: Change to return image_out


def gradient_angle(Ix, Iy):
    """Compute angle (orientation) image given X and Y gradients.

    Parameters
    ----------
        Ix (numpy.array): image gradient in X direction
        Iy (numpy.array): image gradient in Y direction, same size and type as Ix

    Returns
    -------
        angle (numpy.array): gradient angle image, same shape as Ix and Iy. Values must be in degrees [0.0, 360)
    """

    # TODO: Your code here
    pass  # TODO: Change to return angle


def get_draw_keypoints(points, R, angle, size, octave=0):
    """Create OpenCV KeyPoint objects given interest points, response and angle images.
    See cv2.KeyPoint and cv2.drawKeypoint

    Parameters
    ----------
        points (numpy.array): interest points (e.g. corners), array of [x, y] coordinates
        R (numpy.array): floating-point response map, e.g. output from the Harris detector
        angle (numpy.array): gradient angle (orientation) image, each value in degrees [0, 360).
                             Keep in mind this is a [row, col] array. To obtaing the correct
                             angle value you should use angle[y, x]
        size (float): fixed _size parameter to pass to cv2.KeyPoint() for all points
        octave (int): fixed _octave parameter to pass to cv2.KeyPoint() for all points.
                      This parameter can be left as 0.

    Returns
    -------
        keypoints (list): a sequence of cv2.KeyPoint objects
        image_out (numpy.array): output image with keypoints drawn on it
    """

    # TODO: Your code here
    # Note: You should be able to plot the keypoints using cv2.drawKeypoints() in OpenCV 2.4.9+
    pass  # TODO: Change to return keypoints, image_out


def get_descriptors(image, keypoints):
    """Extract feature descriptors from image at each keypoint.

    Parameters
    ----------
        keypoints (list): a sequence of cv2.KeyPoint objects

    Returns
    -------
        descriptors (numpy.array): 2D array of shape (len(keypoints), 32)
        new_kp (list): keypoints from orb.compute
    """

    image_norm = np.zeros(image.shape, dtype=np.uint8)
    cv2.normalize(image, dst=image_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # TODO: Your code here. Work with image_norm.
    # Note: You can use OpenCV's ORB.compute() method to extract descriptors, or write your own!
    pass  # TODO: Change to return descriptors, new_kp


def match_descriptors(desc1, desc2):
    """Match feature descriptors obtained from two images.
    Use cv2.NORM_HAMMING and cross check for cv2.BFMatcher.
    Finally return the matches sorted by distance.

    Parameters
    ----------
        desc1 (numpy.array): descriptors from image 1, as returned by ORB.compute()
        desc2 (numpy.array): descriptors from image 2, same format as desc1

    Returns
    -------
        matches (list): a sequence (list) of cv2.DMatch objects containing corresponding descriptor indices
    """

    # TODO: Your code here
    # Note: You can use OpenCV's descriptor matchers, or write your own!
    #       Make sure you use Hamming Normalization to match the autograder.
    pass  # TODO: Change to return matches


def draw_matches(image1, image2, kp1, kp2, matches):
    """Show matches by drawing lines connecting corresponding keypoints.
    Results must be presented joining the input images side by side
    (use make_image_pair()).

    Parameters
    ----------
        image1 (numpy.array): first image
        image2 (numpy.array): second image, same type as first
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2
        matches (list): list of matching keypoint index pairs (as cv2.DMatch objects)

    Returns
    -------
        image_out (numpy.array): image1 and image2 joined side-by-side with matching lines;
                                 color image (BGR), uint8, values in [0, 255]
    """

    # TODO: Your code here
    # Note: DO NOT use OpenCV's match drawing function(s)! Write your own.
    pass  # TODO: Change to return image_out


def compute_translation_RANSAC(kp1, kp2, matches, thresh=0):
    """Compute best translation vector using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2
        matches (list): list of matches (as cv2.DMatch objects)
        thresh (float): offset tolerance

    Returns
    -------
        translation (numpy.array): translation/offset vector <x, y>, array of shape (2, 1)
        good_matches (list): consensus set of matches that agree with this translation
    """

    # TODO: Your code here
    pass  # TODO: Change to return translation, good_matches


def compute_similarity_RANSAC(kp1, kp2, matches, thresh=0):
    """Compute best similarity transform using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2
        matches (list): list of matches (as cv2.DMatch objects)
        thresh (float): offset tolerance in pixels

    Returns
    -------
        transform (numpy.array): similarity transform matrix, NumPy array of shape (2, 3)
        good_matches (list): consensus set of matches that agree with this transform
    """

    # TODO: Your code here
    pass  # TODO: Change to return transform, good_matches


def compute_affine_RANSAC(kp1, kp2, matches, thresh=0):
    """Compute best affine transform using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2
        matches (list): list of matches (as cv2.DMatch objects)
        thresh (float): offset tolerance in pixels

    Returns
    -------
        transform (numpy.array): affine transform matrix, NumPy array of shape (2, 3)
        good_matches (list): consensus set of matches that agree with this transform
    """

    # TODO: Your code here
    pass  # TODO: Change to return transform, good_matches


def warp_img(img_a, img_b, transform):
    """ Warp image B using a transformation matrix.
    Keep in mind:
    - Write your own warping function. No OpenCV functions are allowed
    - If you see several black pixels (dots) in your image, it means you are not
      implementing backwards warping.
    - If line segments do not seem straight you can apply interpolation methods.
      https://en.wikipedia.org/wiki/Interpolation
      https://en.wikipedia.org/wiki/Bilinear_interpolation

    Parameters
    ----------
        img_a (numpy.array): reference image
        img_b (numpy.array): image to be warped
        transform (numpy.array): transform matrix, array of shape (2, 3)

    Returns
    -------
        warpedB (numpy.array): warped image
        overlay (numpy.array): reference and warped image overlaid. Copy the reference
                               image in the red channel and the warped image in the
                               green channel
    """

    # TODO: Your code here
    pass  # TODO: Change to return overlay, warpedB
