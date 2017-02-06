"""Problem Set 2: Edges and Lines."""

import math
import numpy as np
import cv2


def hough_lines_acc(img_edges, rho_res=1, theta_res=np.pi/180):
    """ Creates and returns a Hough accumulator array by computing the Hough Transform for lines on an
    edge image.

    This method defines the dimensions of the output Hough array using the rho and theta resolution
    parameters. The Hough accumulator is a 2-D array where its rows and columns represent the index
    values of the vectors rho and theta respectively. The length of each dimension is set by the
    resolution parameters. For example: if rho is a vector of values that are in [a_0, a_1, a_2, ... a_n],
    and rho_res = 1, rho should remain as [a_0, a_1, a_2, ... , a_n]. If rho_res = 2, then rho would
    be half its length i.e [a_0, a_2, a_4, ... , a_n] (assuming n is even). The same description applies
    to theta_res and the output vector theta. These two parameters define the size of each bin in
    the Hough array.

    Note that indexing using negative numbers will result in calling index values starting from
    the end. For example, if b = [0, 1, 2, 3, 4] calling b[-2] will return 3.

    Args:
        img_edges (numpy.array): edge image (every nonzero value is considered an edge).
        rho_res (int): rho resolution (in pixels).
        theta_res (float): theta resolution (in degrees converted to radians i.e 1 deg = pi/180).

    Returns:
        tuple: Three-element tuple containing:
               H (numpy.array): Hough accumulator array.
               rho (numpy.array): vector of rho values, one for each row of H
               theta (numpy.array): vector of theta values, one for each column of H.
    """
    row_size, col_size = img_edges.shape
    # construct theta array [0, pi)
    theta = np.linspace(0, 180 * theta_res, 180, endpoint=False)
    # get the maximum rho possible
    rho_max = int(math.ceil(math.sqrt(row_size**2 + col_size**2)))
    # construct the rho array
    rho = np.linspace(-rho_max * rho_res, rho_max * rho_res, rho_max * 2, endpoint=False)
    # initialize H[d, theta] = 0
    H = np.zeros((rho.size, theta.size))

    rows, cols = np.where(img_edges == 255)
    for index in xrange(len(rows)):
        y, x = rows[index], cols[index]
        d_list = (x * np.cos(theta)) + (y * np.sin(theta))  # maybe negative
        for i, d in enumerate(d_list):
            rho_index = np.argmin(np.abs(rho - d))
            H[rho_index, i] += 1

    return H, rho, theta


def hough_peaks(H, hough_threshold, nhood_delta, rows=None, cols=None):
    """Returns the best peaks in a Hough Accumulator array.

    This function selects the pixels with the highest intensity values in an area and returns an array
    with the row and column indices that correspond to a local maxima. This search will only look at pixel
    values that are greater than or equal to hough_threshold.

    Part of this function performs a non-maxima suppression using the parameter nhood_delta which will
    indicate the area that a local maxima covers. This means that any other pixels, with a non-zero values,
    that are inside this area will not be counted as a peak eliminating possible duplicates. The
    neighborhood is a rectangular area of shape nhood_delta[0] * 2 by nhood_delta[1] * 2.

    When working with hough lines, you may need to use the true value of the rows and columns to suppress
    duplicate values due to aliasing. You can use the rows and cols parameters to access the true value of
    for rho and theta at a specific peak.

    Args:
        H (numpy.array): Hough accumulator array.
        hough_threshold (int): minimum pixel intensity value in the accumulator array to search for peaks
        nhood_delta (tuple): a pair of integers indicating the distance in the row and
                             column indices deltas over which non-maximal suppression should take place.
        rows (numpy.array): array with values that map H rows. Default set to None.
        cols (numpy.array): array with values that map H columns. Default set to None.

    Returns:
        numpy.array: Output array of shape Q x 2 where each row is a [row_id, col_id] pair
                     where the peaks are in the H array and Q is the number of the peaks found in H.
    """
    # In order to standardize the range of hough_threshold values let's work with a normalized version of H.
    H_norm = cv2.normalize(H.copy(), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # get the possible peaks index given the threshold
    p_rows, p_cols = np.where(H_norm >= hough_threshold)
    # get the peak values
    values = H_norm[p_rows, p_cols]
    # arrange matrix in the order of [[x, y, value], [x, y, value]]
    possible_peaks = np.transpose(np.concatenate((p_rows, p_cols, values)).reshape(3, values.size))
    # sort descending order by value of the most voted peak
    possible_peaks = possible_peaks[possible_peaks[:, 2].argsort()[::-1]]

    # create a clone copy of H
    newH = np.copy(H_norm)
    new_possible_peaks = []

    # Once you have all the detected peaks, you can eliminate the ones that represent
    # the same line. This will only be helpful when working with Hough lines.
    # The autograder does not pass these parameters when using a Hough circles array because it is not
    # needed. You can opt out from implementing this part, make sure you comment it out or delete it.
    # if rows is not None and cols is not None:
    cols_size, rows_size = H.shape
    # Aliasing Suppression.
    for peak in possible_peaks:
        x = int(peak[0])
        y = int(peak[1])
        if newH[x, y] == peak[2]:
            x0 = x - nhood_delta[1] if x >= nhood_delta[1] else 0
            x1 = x + nhood_delta[1] if x <= (cols_size - nhood_delta[1]) else cols_size
            y0 = y - nhood_delta[0] if y >= nhood_delta[1] else 0
            y1 = y + nhood_delta[0] if y <= (rows_size - nhood_delta[0]) else rows_size
            # set vaues around the peak to 0
            newH[x0:x1, y0:y1] = 0
            # restore the peak value
            newH[x, y] = peak[2]
            # only interested in the peaks that are not replaced
            new_possible_peaks.append(peak)

    possible_peaks = np.array(new_possible_peaks)

    # only return Q pairs
    peaks = possible_peaks[:, :2].astype(np.int) if len(possible_peaks) > 0 else np.array([], dtype=np.int)
    return peaks


def hough_circles_acc(img_orig, img_edges, radius, point_plus=True):
    """Returns a Hough accumulator array using the Hough Transform for circles.

    This function implements two methods: 'single point' and 'point plus'. Refer to the problem set
    instructions and the course lectures for more information about them.

    For simplicity purposes, this function returns an array of the same dimensions as img_edges.
    This means each bin corresponds to one pixel (there are no changes to the grid discretization).

    Note that the 'point plus' method requires gradient images in X and Y (see cv2.Sobel) using
    img_orig to perform the voting.

    Args:
        img_orig (numpy.array): original image.
        img_edges (numpy.array): edge image (every nonzero value is considered an edge).
        radius (int): radius value to look for.
        point_plus (bool): flag that allows to choose between 'single point' or 'point plus'.

    Returns:
        numpy.array: Hough accumulator array.
    """

    row_size, col_size = img_edges.shape
    # initialize H[a, b] = 0
    H = np.zeros((row_size, col_size))

    rows, cols = np.where(img_edges == 255)

    if not point_plus:
        theta_res = np.pi/180
        theta = np.linspace(0, 360 * theta_res, 360, endpoint=False)

        for index in xrange(len(rows)):
            y, x = rows[index], cols[index]
            a_list = np.round(y + radius * np.sin(theta)).astype(np.int)
            b_list = np.round(x - radius * np.cos(theta)).astype(np.int)
            for j in xrange(len(a_list)):
                a, b = a_list[j], b_list[j]
                if 0 <= a < row_size and 0 <= b < col_size:
                    H[a, b] += 1
    elif point_plus:
        def get_bin_position(a, b, k=1):
            y0 = a - k if a >= k else 0
            y1 = a + k + 1
            x0 = b - k if b >= k else 0
            x1 = b + k + 1
            return (x0, x1), (y0, y1)

        grad_x = cv2.Sobel(img_orig, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_orig, cv2.CV_64F, 0, 1, ksize=3)

        for index in xrange(len(rows)):
            y, x = rows[index], cols[index]
            dx = grad_x[y, x]
            dy = grad_y[y, x]
            # get the theta
            t = math.atan2(dy, dx)
            # vote in both direction in and out the center
            a1 = int(np.round(y + radius * math.sin(t)))  # row_id
            b1 = int(np.round(x + radius * math.cos(t)))  # col_id
            a2 = int(np.round(y - radius * math.sin(t)))  # row_id
            b2 = int(np.round(x - radius * math.cos(t)))  # col_id
            if 0 <= a1 < row_size and 0 <= b1 < col_size:
                (x0, x1), (y0, y1) = get_bin_position(a1, b1)
                H[y0:y1, x0:x1] += 1
                H[a1, b1] += 10
            if 0 <= a1 < row_size and 0 <= b2 < col_size:
                (x0, x1), (y0, y1) = get_bin_position(a1, b2)
                H[y0:y1, x0:x1] += 1
                H[a1, b2] += 10
            if 0 <= a2 < row_size and 0 <= b1 < col_size:
                (x0, x1), (y0, y1) = get_bin_position(a2, b1)
                H[y0:y1, x0:x1] += 1
                H[a2, b1] += 10
            if 0 <= a2 < row_size and 0 <= b2 < col_size:
                (x0, x1), (y0, y1) = get_bin_position(a2, b2)
                H[y0:y1, x0:x1] += 1
                H[a2, b2] += 10

    return H


def find_circles(img_orig, edge_img, radii, hough_threshold, nhood_delta):
    """Finds circles in the input edge image using the Hough transform and the point plus gradient
    method.

    In this method you will call both hough_circles_acc and hough_peaks.

    The goal here is to call hough_circles_acc iterating over the values in 'radii'. A Hough accumulator
    is generated for each radius value and the respective peaks are identified. It is recommended that
    the peaks from all radii are stored with their respective vote value. That way you can identify which
    are true peaks and discard false positives.

    Args:
        img_orig (numpy.array): original image. Pass this parameter to hough_circles_acc.
        edge_img (numpy.array): edge image (every nonzero value is considered an edge).
        radii (list): list of radii values to search for.
        hough_threshold (int): minimum pixel intensity value in the accumulator array to
                               search for peaks. Pass this value to hough_peaks.
        nhood_delta (tuple): a pair of integers indicating the distance in the row and
                             column indices deltas over which non-maximal suppression should
                             take place. Pass this value to hough_peaks.

    Returns:
        numpy.array: array with the circles position and radius where each row
                     contains [row_id, col_id, radius]
    """
    row_size, col_size = edge_img.shape
    H = np.zeros((row_size, col_size, 0))
    row_range = np.linspace(0, col_size - 1, col_size)
    col_range = np.linspace(0, row_size - 1, row_size)
    peaks = np.zeros((0, 4))
    circles = np.zeros((0, 4))
    for r_index, r in enumerate(radii):
        # vote for circles given radius r
        h = hough_circles_acc(img_orig, edge_img, radius=r, point_plus=True)
        H = np.concatenate((H, h.reshape((row_size, col_size, 1))), axis=2)
        # get the possible peaks for radius r
        possible_peaks = hough_peaks(h, hough_threshold=hough_threshold, nhood_delta=nhood_delta)
        if len(possible_peaks) > 0:
            values = h[possible_peaks[:, 0], possible_peaks[:, 1]]
            # insert radius to the matrix
            possible_peaks = np.insert(possible_peaks, possible_peaks.shape[1], r_index, axis=1)
            # concat peaks with values
            possible_peaks = np.concatenate((possible_peaks, values.reshape(values.shape[0], 1)), axis=1)
            peaks = np.concatenate((peaks, possible_peaks))

    peaks = peaks[peaks[:, peaks.shape[1]-1].argsort()[::-1]].astype(np.int)

    # Since our H is 3-D, we want to use the most voted circle and remove the other radius
    for peak in peaks:
        x, y, z = peak[:3]
        if H[x, y, z] == peak[3]:
            x0 = x - nhood_delta[1] if x >= nhood_delta[1] else 0
            x1 = x + nhood_delta[1] + 1 if x <= (col_range.size - nhood_delta[1]) else col_range.size
            y0 = y - nhood_delta[0] if y >= nhood_delta[0] else 0
            y1 = y + nhood_delta[0] + 1 if y <= (row_range.size - nhood_delta[0]) else row_range.size
            # set values around the peak to 0
            for i in xrange(len(radii)):
                H[x0:x1, y0:y1, i] = 0
            # restore the peak value
            H[x, y, z] = peak[3]
            # only interested in the peaks that are not replaced
            circles = np.concatenate((circles, [[x, y, radii[z], peak[3]]]))

    return circles[:, :3]
