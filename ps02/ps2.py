"""Problem Set 2: Edges and Lines."""

import math
import numpy as np
import cv2


def hough_lines_acc(img_edges, rho_res=1, theta_res=None):
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

    pass  # TODO: change to return H, rho, theta


def hough_peaks(H, Q, cols, rows, hough_threshold=200, nhood_radii=(5, 5)):
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

    pass  # TODO: change to return peaks


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

    pass  # TODO: change to return H


def find_circles(edge_img, grad_x, grad_y, radii=None, hough_threshold=None, nhood_radii=None):
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

    pass  # TODO: change to return circles




