"""Problem Set 5: Harris, ORB, RANSAC."""

import numpy as np
import cv2


def gradient_x(image):
    """Computes the image gradient in X direction.

    This method returns an image gradient considering the X direction. See cv2.Sobel.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in X direction with values in [-1.0, 1.0].
    """
    gradient = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    return gradient / np.linalg.norm(gradient)


def gradient_y(image):
    """Computes the image gradient in Y direction.

    This method returns an image gradient considering the Y direction. See cv2.Sobel.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in Y direction with values in [-1.0, 1.0].
    """
    gradient = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return gradient / np.linalg.norm(gradient)


def make_image_pair(image1, image2):
    """Adjoins two images side-by-side to make a single new image.

    The output dimensions must take the maximum height from both images for the total height.
    The total width is found by adding the widths of image1 and image2.

    Args:
        image1 (numpy.array): first image, could be grayscale or color (BGR).
                              This array takes the left side of the output image.
        image2 (numpy.array): second image, could be grayscale or color (BGR).
                              This array takes the right side of the output image.

    Returns:
        numpy.array: combination of both images, side-by-side, same type as the input size.
    """
    img1_shape = image1.shape
    img2_shape = image2.shape
    if img1_shape[0] == img2_shape[0]:
        img1 = image1
        img2 = image2
    elif img1_shape[0] > img2_shape[0]:
        img1 = image1
        pad_tuple = tuple([(0, img1_shape[0] - img2_shape[0])] + [(0, 0) for _ in xrange(len(img1_shape) - 1)])
        img2 = np.pad(image2, pad_tuple, 'constant', constant_values=0)
    else:
        img2 = image2
        pad_tuple = tuple([(0, img2_shape[0] - img1_shape[0])] + [(0, 0) for _ in xrange(len(img2_shape) - 1)])
        img1 = np.pad(image1, pad_tuple, 'constant', constant_values=0)
    # TODO: handle combination of grayscale and color images
    return np.concatenate((img1, img2), axis=1)


def harris_response(ix, iy, kernel_dims, alpha):
    """Computes the Harris response map using given image gradients.

    Args:
        ix (numpy.array): image gradient in the X direction with values in [-1.0, 1.0].
        iy (numpy.array): image gradient in the Y direction with the same shape and type as Ix.
        kernel_dims (tuple): 2D windowing kernel dimensions. ie. (3, 3)  (3, 5).
        alpha (float): Harris detector parameter multiplied with the square of trace.

    Returns:
        numpy.array: Harris response map, same size as inputs, floating-point.
    """
    # uniform kernel
    w = np.ones(kernel_dims) / (kernel_dims[0] * kernel_dims[1])

    # # gaussian kernel
    # kx, ky = kernel
    # sigma = 3  # np.max([kx, ky])
    # x_interval = (2 * sigma + 1.) / kx
    # y_interval = (2 * sigma + 1.) / ky
    # x_space = np.linspace(-sigma - x_interval/2., sigma + x_interval/2., kx+1)
    # y_space = np.linspace(-sigma - y_interval/2., sigma + y_interval/2., ky+1)
    # # x_space = np.linspace(-kx, kx, kx+1)
    # # y_space = np.linspace(-ky, ky, ky+1)
    # x_diff = np.diff(st.norm.cdf(x_space))
    # y_diff = np.diff(st.norm.cdf(y_space))
    # kernel_raw = np.sqrt(np.outer(x_diff, y_diff))
    # w = kernel_raw / kernel_raw.sum()
    # w_out = cv2.normalize(w, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    M00 = cv2.filter2D(ix * ix, -1, w)
    M01 = M10 = cv2.filter2D(ix * iy, -1, w)
    M11 = cv2.filter2D(iy * iy, -1, w)

    det_M = (M00 * M11) - (M01 * M10)
    trace_M = M00 + M11

    R = det_M - (alpha * (trace_M ** 2))
    return R


def find_corners(r_map, threshold, radius):
    """Finds corners in a given response map.

    This method uses a circular region to define the non-maxima suppression area. For example,
    let c1 be a corner representing a peak in the Harris response map, any corners in the area
    determined by the circle of radius 'radius' centered in c1 should not be returned in the
    peaks array.

    Make sure you account for duplicate and overlapping points.

    Args:
        r_map (numpy.array): floating-point response map, e.g. output from the Harris detector.
        threshold (float): value between 0.0 and 1.0. Response values less than this should
                           not be considered plausible corners.
        radius (int): radius of circular region for non-maximal suppression.

    Returns:
        numpy.array: peaks found in response map R, each row must be defined as [x, y]. Array
                     size must be N x 2, where N are the number of points found.
    """

    # Normalize R
    r_map_norm = cv2.normalize(r_map, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # Do not modify the code above. Continue working with r_map_norm.

    if threshold < 0. or threshold > 1.:
        return np.array([])

    # get the possible peaks indices given the threshold
    p_rows, p_cols = np.where(r_map_norm >= threshold)
    # get the peak values
    values = r_map_norm[p_rows, p_cols]
    # arrange matrix in the order of [[y, x, value], [y, x, value]]
    possible_peaks = np.vstack((p_rows, p_cols, values)).T
    # sort descending order by value of the most bright
    possible_peaks = possible_peaks[possible_peaks[:, 2].argsort()[::-1]]

    new_r_map_norm = r_map_norm.copy()
    peaks = []

    for peak in possible_peaks:
        if new_r_map_norm[int(peak[0]), int(peak[1])] == peak[2]:
            y0 = int(peak[0] - radius - 1 if peak[0] >= radius else 0)
            y1 = int(peak[0] + radius + 2 if peak[0] <= r_map.shape[0] - radius else r_map.shape[0])
            x0 = int(peak[1] - radius - 1 if peak[1] >= radius else 0)
            x1 = int(peak[1] + radius + 2 if peak[1] <= r_map.shape[1] - radius else r_map.shape[1])
            new_r_map_norm[y0:y1, x0:x1] = 0
            new_r_map_norm[int(peak[0]), int(peak[1])] = peak[2]
            peaks.append(peak)

    peaks = np.array(peaks)[:, :2][:, ::-1]
    return peaks.astype(np.int)


def draw_corners(image, corners):
    """Draws corners on (a copy of) the given image.

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        corners (numpy.array): peaks found in response map R, as a sequence of [x, y] coordinates.
                               Array size must be N x 2, where N are the number of points found.

    Returns:
        numpy.array: copy of the input image with corners drawn on it, in color (BGR).
    """
    image_norm = cv2.normalize(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    image_out = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2BGR)

    for corner in corners:
        cv2.circle(image_out, tuple(corner), 2, (0, 255, 0), -1)
    return image_out


def gradient_angle(ix, iy):
    """Computes the angle (orientation) image given the X and Y gradients.

    Args:
        ix (numpy.array): image gradient in X direction.
        iy (numpy.array): image gradient in Y direction, same size and type as Ix

    Returns:
        numpy.array: gradient angle image, same shape as ix and iy. Values must be in degrees [0.0, 360).
    """
    angle = np.arctan2(iy, ix) / np.pi * 180
    return angle


def get_keypoints(points, angle, size, octave=0):
    """Creates OpenCV KeyPoint objects given interest points, response map, and angle images.

    See cv2.KeyPoint and cv2.drawKeypoint.

    Args:
        points (numpy.array): interest points (e.g. corners), array of [x, y] coordinates.
        angle (numpy.array): gradient angle (orientation) image, each value in degrees [0, 360).
                             Keep in mind this is a [row, col] array. To obtain the correct
                             angle value you should use angle[y, x].
        size (float): fixed _size parameter to pass to cv2.KeyPoint() for all points.
        octave (int): fixed _octave parameter to pass to cv2.KeyPoint() for all points.
                      This parameter can be left as 0.

    Returns:
        keypoints (list): a sequence of cv2.KeyPoint objects
    """

    # Note: You should be able to plot the keypoints using cv2.drawKeypoints() in OpenCV 2.4.9+
    keypoints = []
    for point in points:
        x, y = point
        keypoint = cv2.KeyPoint(x=x, y=y, _size=size, _angle=angle[y, x], _octave=octave)
        keypoints.append(keypoint)

    return keypoints


def get_descriptors(image, keypoints):
    """Extracts feature descriptors from the image at each keypoint.

    This function finds descriptors following the methods used in cv2.ORB. You are allowed to
    use such function or write your own.

    Args:
        image (numpy.array): input image where the descriptors will be computed from.
        keypoints (list): a sequence of cv2.KeyPoint objects.

    Returns:
        tuple: 2-element tuple containing:
            descriptors (numpy.array): 2D array of shape (len(keypoints), 32).
            new_kp (list): keypoints from ORB.compute().
    """

    # Normalize image
    image_norm = cv2.normalize(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Do not modify the code above. Continue working with r_norm.

    # Note: You can use OpenCV's ORB.compute() method to extract descriptors, or write your own!
    orb = cv2.ORB()
    keypoints, descriptors = orb.compute(image_norm, keypoints)
    return descriptors, keypoints


def match_descriptors(desc1, desc2):
    """Matches feature descriptors obtained from two images.

    Use cv2.NORM_HAMMING and cross check for cv2.BFMatcher. Return the matches sorted by distance.

    Args:
        desc1 (numpy.array): descriptors from image 1, as returned by ORB.compute().
        desc2 (numpy.array): descriptors from image 2, same format as desc1.

    Returns:
        list: a sequence (list) of cv2.DMatch objects containing corresponding descriptor indices.
    """

    # Note: You can use OpenCV's descriptor matchers, or write your own!
    #       Make sure you use Hamming Normalization to match the autograder.
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bfm.match(desc1, desc2)
    return matches


def draw_matches(image1, image2, kp1, kp2, matches):
    """Shows matches by drawing lines connecting corresponding keypoints.

    Results must be presented joining the input images side by side (use make_image_pair()).

    OpenCV's match drawing function(s) are not allowed.

    Args:
        image1 (numpy.array): first image
        image2 (numpy.array): second image, same type as first
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2
        matches (list): list of matching keypoint index pairs (as cv2.DMatch objects)

    Returns:
        numpy.array: image1 and image2 joined side-by-side with matching lines;
                     color image (BGR), uint8, values in [0, 255].
    """

    # Note: DO NOT use OpenCV's match drawing function(s)! Write your own.
    image1_norm = cv2.normalize(image1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image2_norm = cv2.normalize(image2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image_out = make_image_pair(image1_norm, image2_norm)
    image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    for match in matches:
        kp1_idx = match.queryIdx
        kp2_idx = match.trainIdx
        x1, y1 = np.array(kp1[kp1_idx].pt, dtype=np.int)
        x2, y2 = np.array(kp2[kp2_idx].pt, dtype=np.int)

        cv2.line(image_out, (x1, y1), (x2 + image1.shape[1], y2), color=(0, 255, 0))

    return image_out


def calculate_ssd(actual, prediction):
    return np.sqrt(np.sum(np.square(prediction - actual)) / len(actual))


class CustomMatch():
    def __init__(self, match, p1, p2):
        self.queryIdx = match.queryIdx
        self.trainIdx = match.trainIdx
        self.queryPt = p1
        self.trainPt = p2


def compute_translation_RANSAC(kp1, kp2, matches, thresh):
    """Computes the best translation vector using RANSAC given keypoint matches.

    Args:
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1.
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2.
        matches (list): list of matches (as cv2.DMatch objects).
        thresh (float): offset tolerance in pixels which decides if a match forms part of
                        the consensus. This value can be seen as a minimum delta allowed
                        between point components.

    Returns:
        tuple: 2-element tuple containing:
            translation (numpy.array): translation/offset vector <x, y>, array of shape (2, 1).
            good_matches (list): consensus set of matches that agree with this translation.
    """

    # Note: this function must use the RANSAC method. If you implement any non-RANSAC approach
    # (i.e. brute-force) you will not get credit for either the autograder tests or the report
    # sections that depend of this function.
    translation = np.zeros((2, 1))
    good_matches = []
    s = 1

    if len(matches) < s:
        return translation, good_matches

    N = np.inf
    sample_count = 0
    e = 1.
    p = .99

    while N > sample_count:
        sample = np.random.choice(matches, s, replace=False)[0]
        t1 = np.array(kp1[sample.queryIdx].pt)
        t2 = np.array(kp2[sample.trainIdx].pt)
        _translation = t2 - t1

        inliers = []
        t_avg = np.zeros((0, 2))
        for match in matches:
            p1 = np.array(kp1[match.queryIdx].pt)
            p2 = np.array(kp2[match.trainIdx].pt)
            d = p2 - p1

            # difference between d and _translation
            ssd = calculate_ssd(d, _translation)
            if ssd <= thresh:
                inliers.append(CustomMatch(match, p1, p2))
                t_avg = np.concatenate((t_avg, [d]))

        if len(good_matches) < len(inliers) or len(good_matches) == 0:
            good_matches = inliers
            translation = np.average(t_avg, axis=0).reshape(2, 1)

        e0 = 1 - (len(inliers) / float(len(matches)))

        if e0 < e:
            e = e0
            N = np.log(1 - p) / np.log(1 - np.power(1 - e, s))

        sample_count += 1

    print len(good_matches) / float(len(matches))

    return translation, good_matches


def compute_similarity_RANSAC(kp1, kp2, matches, thresh):
    """Computes the best similarity transform using RANSAC given keypoint matches.

    Args:
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1.
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2.
        matches (list): list of matches (as cv2.DMatch objects).
        thresh (float): offset tolerance in pixels which decides if a match forms part of
                        the consensus. This value can be seen as a minimum delta allowed
                        between point components.

    Returns:
        tuple: 2-element tuple containing:
            m (numpy.array): similarity transform matrix of shape (2, 3).
            good_matches (list): consensus set of matches that agree with this transformation.
    """

    # Note: this function must use the RANSAC method. If you implement any non-RANSAC approach
    # (i.e. brute-force) you will not get credit for either the autograder tests or the report
    # sections that depend of this function.
    transform = np.zeros((2, 3))
    good_matches = []
    s = 2

    if len(matches) < s:
        return transform, good_matches

    N = np.inf
    sample_count = 0
    e = 1.
    p = .99

    def calculate_transform(kp1, kp2, samples):
        A = np.zeros((0, 4))
        B = np.zeros((0, 1))
        for sample in samples:
            u, v = np.array(kp1[sample.queryIdx].pt)
            u_p, v_p = np.array(kp2[sample.trainIdx].pt)
            A = np.concatenate((A, [[u, -v, 1, 0],
                                    [v,  u, 0, 1]]))
            B = np.concatenate((B, [[u_p],
                                    [v_p]]))
        a, b, c, d = np.linalg.lstsq(A, B)[0]
        transform = np.array([[a, -b, c],
                              [b,  a, d]]).reshape(2, 3)
        return transform

    while N > sample_count:
        samples = np.random.choice(matches, s, replace=False)
        _transform = calculate_transform(kp1, kp2, samples)

        inliers = []
        for match in matches:
            p1 = np.array(kp1[match.queryIdx].pt)
            p2 = np.array(kp2[match.trainIdx].pt)
            h_p1 = np.concatenate((p1, [1])).reshape(3, 1)
            ep = np.dot(_transform, h_p1)

            # difference between p2 and ep
            ssd = calculate_ssd(p2, ep.reshape(ep.shape[0],))
            if ssd <= thresh:
                inliers.append(CustomMatch(match, p1, p2))

        if len(good_matches) < len(inliers) or len(good_matches) == 0:
            good_matches = inliers

        e0 = 1 - (len(inliers) / float(len(matches)))

        if e0 < e:
            e = e0
            N = np.log(1 - p) / np.log(1 - np.power(1 - e, s))

        sample_count += 1

    transform = calculate_transform(kp1, kp2, good_matches)

    print len(good_matches) / float(len(matches))

    return transform, good_matches


def compute_affine_RANSAC(kp1, kp2, matches, thresh):
    """ Compute the best affine transform using RANSAC given keypoint matches.

    Args:
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2
        matches (list): list of matches (as cv2.DMatch objects)
        thresh (float): offset tolerance in pixels which decides if a match forms part of
                        the consensus. This value can be seen as a minimum delta allowed
                        between point components.

    Returns:
        tuple: 2-element tuple containing:
            m (numpy.array): affine transform matrix of shape (2, 3)
            good_matches (list): consensus set of matches that agree with this transformation.
    """

    # Note: this function must use the RANSAC method. If you implement any non-RANSAC approach
    # (i.e. brute-force) you will not get credit for either the autograder tests or the report
    # sections that depend of this function.
    transform = np.zeros((2, 3))
    good_matches = []
    s = 3

    if len(matches) < s:
        return transform, good_matches

    N = np.inf
    sample_count = 0
    e = 1.
    p = .99

    def calculate_transform(kp1, kp2, samples):
        A = np.zeros((0, 6))
        B = np.zeros((0, 1))
        for sample in samples:
            u, v = np.array(kp1[sample.queryIdx].pt)
            u_p, v_p = np.array(kp2[sample.trainIdx].pt)
            A = np.concatenate((A, [[u, v, 1, 0, 0, 0],
                                    [0, 0, 0, u, v, 1]]))
            B = np.concatenate((B, [[u_p],
                                    [v_p]]))
        a, b, c, d, e, f = np.linalg.lstsq(A, B)[0]
        transform = np.array([[a, b, c],
                              [d, e, f]]).reshape(2, 3)
        return transform

    while N > sample_count:
        samples = np.random.choice(matches, s, replace=False)
        _transform = calculate_transform(kp1, kp2, samples)

        inliers = []
        for match in matches:
            p1 = np.array(kp1[match.queryIdx].pt)
            p2 = np.array(kp2[match.trainIdx].pt)
            h_p1 = np.concatenate((p1, [1])).reshape(3, 1)
            ep = np.dot(_transform, h_p1)

            # difference between p2 and ep
            ssd = calculate_ssd(p2, ep.reshape(ep.shape[0],))
            if ssd <= thresh:
                inliers.append(CustomMatch(match, p1, p2))

        if len(good_matches) < len(inliers) or len(good_matches) == 0:
            good_matches = inliers

        e0 = 1 - (len(inliers) / float(len(matches)))

        if e0 < e:
            e = e0
            N = np.log(1 - p) / np.log(1 - np.power(1 - e, s))

        sample_count += 1

    transform = calculate_transform(kp1, kp2, good_matches)

    print len(good_matches) / float(len(matches))

    return transform, good_matches


def warp_img(img_a, img_b, m):
    """Warps image B using a transformation matrix.

    Keep in mind:
    - Write your own warping function. No OpenCV functions are allowed.
    - If you see several black pixels (dots) in your image, it means you are not
      implementing backwards warping.
    - If line segments do not seem straight you can apply interpolation methods.
      https://en.wikipedia.org/wiki/Interpolation
      https://en.wikipedia.org/wiki/Bilinear_interpolation

    Args:
        img_a (numpy.array): reference image.
        img_b (numpy.array): image to be warped.
        m (numpy.array): transformation matrix, array of shape (2, 3).

    Returns:
        tuple: 2-element tuple containing:
            warpedB (numpy.array): warped image.
            overlay (numpy.array): reference and warped image overlaid. Copy the reference
                                   image in the red channel and the warped image in the
                                   green channel
    """

    # Note: Write your own warping function. No OpenCV warping functions are allowed.
    warpedB = np.zeros_like(img_b)
    for x in xrange(img_b.shape[1]):
        for y in xrange(img_b.shape[0]):
            h_p = np.array([[x], [y], [1]])
            i, j = np.dot(m, h_p).reshape(2,)
            if j < 0 or j >= img_b.shape[0] - 1:
                warpedB[y, x] = 0.
            elif i < 0 or i >= img_b.shape[1] - 1:
                warpedB[y, x] = 0.
            else:
                a = i - np.floor(i)
                b = j - np.floor(j)
                i = int(i)
                j = int(j)
                # bilinear interpolation
                warpedB[y, x] = (1 - a) * (1 - b) * img_b[j, i]
                warpedB[y, x] += a * (1 - b) * img_b[j, i + 1]
                warpedB[y, x] += a * b * img_b[j + 1, i + 1]
                warpedB[y, x] += (1 - a) * b * img_b[j + 1, i]

    warpedB = cv2.normalize(warpedB, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_a_norm = cv2.normalize(img_a, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    overlay = np.zeros(list(img_a.shape) + [3])
    overlay[..., 2] = img_a_norm  # red channel
    overlay[..., 1] = warpedB  # green channel

    return warpedB, overlay
