"""Problem Set 2: Edges and Lines."""

import math
import numpy as np
import cv2


def hough_lines_acc(img_edges, rho_res=1, theta_res=np.pi/180):
    """Compute Hough Transform for lines on edge image.

    Parameters
    ----------
        img_edges (numpy.array): edge image (every nonzero value is considered an edge)
        rho_res (int): rho resolution (in pixels)
        theta_res (float): theta resolution (in radians)

    Returns
    -------
        H (numpy.array): Hough accumulator array
        rho (numpy.array): vector of rho values, one for each row of H
        theta (numpy.array): vector of theta values in the range [0,pi), one for each column of H
    """
    rows, cols = img_edges.shape
    # construct theta array [0, pi)
    theta = np.array([i * theta_res for i in xrange(0, 180)])
    # get the maximum rho possible
    rho_max = int(math.ceil(math.sqrt(rows**2 + cols**2)))
    # construct the rho array
    rho = np.array([i * rho_res for i in xrange(-rho_max, rho_max)])
    # initialize H[d, theta] = 0
    H = np.zeros((rho.size, theta.size))

    for y in range(rows):
        for x in range(cols):
            # for each edge point
            if img_edges[y, x] == 255:
                for i, t in enumerate(theta):
                    d = x * math.cos(t) + y * math.sin(t)  # maybe negative
                    rho_index = np.argmin(np.abs(rho - d))
                    H[rho_index, i] += 1

    return H, rho, theta


def hough_peaks(H, Q, cols, rows, hough_threshold=None, nhood_radii=(0, 0)):
    """

    Parameters
    ----------
        H (numpy.array): Hough accumulator array
        Q (int): number of peaks to find (max)
        cols (numpy.array): 1D array with column values, ie. theta = [-pi : pi]
        rows (numpy.array): 1D array with row values, ie. rho = [-x : x]
        hough_threshold (float): minimum value in accumulator array for peaks
        nhood_radii (tuple): a pair of integers indicating the distance in the rho and
                    theta directions over which non-maximal suppression should take place

    Returns
    -------
        peaks (numpy.array): A matrix where each row is a (col_id, row_id) pair where the peaks are.

    """
    # get the possible peaks index given the threshold
    p_rows, p_cols = np.where(H >= hough_threshold)
    # get the peak values
    values = H[p_rows, p_cols]
    # arrange matrix in the order of [[x, y, value], [x, y, value]]
    possible_peaks = np.transpose(np.concatenate((p_rows, p_cols, values)).reshape(3, values.size))
    # sort descending order by value of the most voted peak
    possible_peaks = possible_peaks[possible_peaks[:, 2].argsort()[::-1]]

    # create a clone copy of H
    newH = np.copy(H)
    # get the padding to the left and right of the peak index
    left_padding = nhood_radii[0] / 2
    right_padding = nhood_radii[1] / 2
    new_possible_peaks = []

    # update the newH so that the values around a peak is 0
    for peak in possible_peaks:
        if newH[peak[0], peak[1]] == peak[2]:
            x0 = peak[0] - left_padding if peak[0] >= left_padding else 0
            x1 = peak[0] + right_padding + 1 if peak[0] <= (cols.size - right_padding) else cols.size
            y0 = peak[1] - left_padding if peak[1] >= left_padding else 0
            y1 = peak[1] + right_padding + 1 if peak[1] <= (rows.size - right_padding) else rows.size
            # set values around the peak to 0
            newH[x0:x1, y0:y1] = 0
            # restore the peak value
            newH[peak[0], peak[1]] = peak[2]
            # only interested in the peaks that are not replaced
            new_possible_peaks.append(peak)

    possible_peaks = np.array(new_possible_peaks, dtype=np.int)

    # only return Q pairs
    peaks = possible_peaks[:, :2][:Q] if len(possible_peaks) > 0 else np.array([], dtype=np.int)
    return peaks


def hough_circles_acc(img_edges, radius=10, method='point plus', grad_x=None, grad_y=None):
    """Compute Hough Transform for lines on edge image.

    Parameters
    ----------
        img_edges (numpy.array): binary edge image
        radius (int): value with the radius to look for
        method (str): string value either 'single point' or 'point plus'
        grad_x (numpy.array): gradient image in X (see cv2.Sobel) used with 'point plus'
        grad_y (numpy.array): gradient image in Y (see cv2.Sobel) used with 'point plus'

    Returns
    -------
        H (numpy.array): Hough accumulator array
    """
    row_size, col_size = img_edges.shape
    # initialize H[a, b] = 0
    H = np.zeros((row_size, col_size))

    if method == 'single point':
        # construct theta array [0, 2*pi)
        theta_res = np.pi/180
        theta = np.array([i * theta_res for i in xrange(0, 360)])

        for y in range(row_size):
            for x in range(col_size):
                # for each edge point
                if img_edges[y, x] == 255:
                    for i, t in enumerate(theta):
                        a = np.round(y + radius * math.sin(t))  # row_id
                        b = np.round(x - radius * math.cos(t))  # col_id
                        if 0 <= a < row_size and 0 <= b < col_size:
                            H[a, b] += 1
    elif method == 'point plus':
        possible_radius = [radius]

        def get_bin_position(a, b, k=1):
            y0 = a - k if a >= k else 0
            y1 = a + k + 1
            x0 = b - k if b >= k else 0
            x1 = b + k + 1
            return (x0, x1), (y0, y1)

        for y in range(row_size):
            for x in range(col_size):
                # for each edge pixel
                if img_edges[y, x] == 255:
                    for r in possible_radius:
                        dx = grad_x[y, x]
                        dy = grad_y[y, x]
                        # get the theta
                        t = math.atan2(dy, dx)
                        # vote in both direction in and out the center
                        a1 = np.round(y + r * math.sin(t))  # row_id
                        b1 = np.round(x + r * math.cos(t))  # col_id
                        a2 = np.round(y - r * math.sin(t))  # row_id
                        b2 = np.round(x - r * math.cos(t))  # col_id
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


def find_circles(edge_img, grad_x, grad_y, radii=None, hough_threshold=None, nhood_radii=(0, 0)):
    """Finds lines in the input edge image using the Hough transform and the point plus
    gradient method. In this method you will call both hough_circles_acc and hough_peaks

    Parameters
    ----------
        edge_img (numpy.array): the input edge image
        grad_x (numpy.array): gradient image in X (see cv2.Sobel)
        grad_y (numpy.array): gradient image in Y (see cv2.Sobel)
        radii: array-like of circle radii to search for
        hough_threshold: minimum fraction of circle perimeter present
        nhood_radii: a pair of integers indicating the distance in the center and
                    radii directions over which non-maximal suppression should take place

    Returns
    -------
        circles (numpy.array): A matrix where each row is a (x, y, r) triple parameterizing a circle.
    """
    row_size, col_size = edge_img.shape
    H = np.zeros((row_size, col_size, 0))
    row_range = np.linspace(0, col_size - 1, col_size)
    col_range = np.linspace(0, row_size - 1, row_size)
    peaks = np.zeros((0, 4))
    circles = np.zeros((0, 4))
    for r_index, r in enumerate(radii):
        # vote for circles given radius r
        h = hough_circles_acc(edge_img, radius=r, method='point plus', grad_x=grad_x, grad_y=grad_y)
        H = np.concatenate((H, h.reshape((row_size, col_size, 1))), axis=2)
        # get the possible peaks for radius r
        possible_peaks = hough_peaks(h, h.size, col_range, row_range, hough_threshold=hough_threshold, nhood_radii=nhood_radii)
        if len(possible_peaks) > 0:
            values = h[possible_peaks[:, 0], possible_peaks[:, 1]]
            # inconsistent peak with find_circles (x, y, r), so flip it
            possible_peaks = possible_peaks[:, ::-1]
            # insert radius to the matrix
            possible_peaks = np.insert(possible_peaks, possible_peaks.shape[1], r_index, axis=1)
            # concat peaks with values
            possible_peaks = np.concatenate((possible_peaks, values.reshape(values.shape[0], 1)), axis=1)
            peaks = np.concatenate((peaks, possible_peaks))

    peaks = peaks[peaks[:, peaks.shape[1]-1].argsort()[::-1]].astype(np.int)

    left_padding = nhood_radii[0] / 2
    right_padding = nhood_radii[1] / 2

    # Since our H is 3-D, we want to use the most voted circle and remove the other radius
    for peak in peaks:
        if H[peak[1], peak[0], peak[2]] == peak[3]:
            x0 = peak[1] - left_padding if peak[1] >= left_padding else 0
            x1 = peak[1] + right_padding + 1 if peak[1] <= (col_range.size - right_padding) else col_range.size
            y0 = peak[0] - left_padding if peak[0] >= left_padding else 0
            y1 = peak[0] + right_padding + 1 if peak[0] <= (row_range.size - right_padding) else row_range.size
            # set values around the peak to 0
            for i in range(len(radii)):
                H[x0:x1, y0:y1, i] = 0
            # restore the peak value
            H[peak[1], peak[0], peak[2]] = peak[3]
            # only interested in the peaks that are not replaced
            circles = np.concatenate((circles, [[peak[0], peak[1], radii[peak[2]], peak[3]]]))

    return circles[:, :3]
