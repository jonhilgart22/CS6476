import numpy as np
import cv2


def solve_least_squares(pts3d, pts2d):
    """Solves for the transformation matrix M that maps each 3D point to corresponding 2D point
    using the least-squares method. See np.linalg.lstsq.

    Args:
        pts3d (numpy.array): 3D global (x, y, z) points of shape (N, 3). Where N is the number of points.
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.

    Returns:
        tuple: two-element tuple containing:
               M (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).
               error (float): sum of squared residuals of all points.
    """
    pass


def project_points(pts3d, m):
    """Projects each 3D point to 2D using the matrix M.

    Args:
        pts3d (numpy.array): 3D global (x, y, z) points of shape (N, 3). Where N is the number of points.
        m (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).

    Returns:
        numpy.array: projected 2D (u, v) points of shape (N, 2). Where N is the same as pts3d.
    """
    pass


def get_residuals(pts2d, pts2d_projected):
    """Computes residual error for each point.

    Args:
        pts2d (numpy.array): observed 2D (u, v) points of shape (N, 2). Where N is the number of points.
        pts2d_projected (numpy.array): 3D global points projected to 2D of shape (N, 2).
                                       Where N is the number of points.

    Returns:
        numpy.array: residual error for each point (L2 distance between each observed and projected 2D points).
                     The array shape must be (N, 1). Where N is the same as in pts2d and pts2d_projected.
    """
    pass


def calibrate_camera(pts3d, pts2d, set_size_k):
    """Finds the best camera projection matrix given corresponding 3D and 2D points.

    Args:
        pts3d (numpy.array): 3D global (x, y, z) points of shape (N, 3). Where N is the number of points.
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.
        set_size_k (int): set of k random points to choose from pts2d.

    Returns:
        tuple: three-element tuple containing:
               bestM (numpy.array): best transformation matrix M of shape (3, 4).
               error (float): sum of squared residuals of all points for bestM.
               avg_residuals (numpy.array): Average residuals array, one row for each iteration.
                                            The array should be of shape (10, 1).
    """
    pass


def get_camera_center(m):
    """Finds the camera global coordinates.

    Args:
        m (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).

    Returns:
        numpy.array: [x, y, z] camera coordinates. Array must be of shape (1, 3).
    """
    pass


def compute_fundamental_matrix(pts2d_1, pts2d_2):
    """Computes the fundamental matrix given corresponding points from 2 images of a scene.

    This function uses the least-squares method, see numpy.linalg.lstsq.

    Args:
        pts2d_1 (numpy.array): 2D points from image 1 of shape (N, 2). Where N is the number of points.
        pts2d_2 (numpy.array): 2D points from image 2 of shape (N, 2). Where N is the number of points.

    Returns:
        numpy.array: array containing the fundamental matrix elements. Array must be of shape (3, 3).
    """
    pass


def reduce_rank(f):
    """Reduces a full rank (3, 3) matrix to rank 2.

    Args:
        f (numpy.array): full rank fundamental matrix. Must be a (3, 3) array.

    Returns:
        numpy.array: rank 2 fundamental matrix. Must be a (3, 3) array.
    """
    pass


def get_epipolar_lines(img1_shape, img2_shape, f, pts2d_1, pts2d_2):
    """Returns epipolar lines using the fundamental matrix and two sets of 2D points.

    Args:
        img1_shape (tuple): image 1 shape (rows, cols)
        img2_shape (tuple): image 2 shape (rows, cols)
        f (numpy.array): Fundamental matrix of shape (3, 3).
        pts2d_1 (numpy.array): 2D points from image 1 of shape (N, 2). Where N is the number of points.
        pts2d_2 (numpy.array): 2D points from image 2 of shape (N, 2). Where N is the number of points.

    Returns:
        tuple: two-element tuple containing:
               epipolar_lines_1 (list): epipolar lines for image 1. Each list element should be
                                        [(x1, y1), (x2, y2)] one for each of the N points.
               epipolar_lines_2 (list): epipolar lines for image 2. Each list element should be
                                        [(x1, y1), (x2, y2)] one for each of the N points.
    """
    pass


def compute_t_matrix(pts2d):
    """Computes the transformation matrix T given corresponding 2D points from an image.

    Args:
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.

    Returns:
        numpy.array: transformation matrix T of shape (3, 3).
    """
    pass


def normalize_points(pts2d, t):
    """Normalizes 2D points.

    Args:
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.
        t (numpy.array): transformation matrix T of shape (3, 3).

    Returns:
        numpy.array: normalized points (N, 2) array.
    """
    pass
