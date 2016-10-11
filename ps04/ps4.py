import numpy as np
import cv2


def solve_least_squares(pts3d, pts2d):
    """ Solve for transformation matrix M that maps each 3D point to corresponding 2D point
    using the least-squares method. See np.linalg.lstsq

    Parameters
    ----------
        pts3d (numpy.array): 3D global (x, y, z) points of shape (N, 3). Where N is the number of points.
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.

    Returns
    -------
        M (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).
        error (float): sum of squared residuals of all points.
    """

    # TODO: Your code here
    pass    # TODO: Change to return M, error


def project_points(pts3d, M):
    """ Project each 3D point to 2D using the matrix M.

    Parameters
    ----------
        pts3d (numpy.array): 3D global (x, y, z) points of shape (N, 3). Where N is the number of points.
        M (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).

    Returns
    -------
        pts2d_projected (numpy.array): projected 2D (u, v) points of shape (N, 2). Where N is the same as pts3d.
    """

    # TODO: Your code here
    pass  # TODO: Change to return pts2d_projected


def get_residuals(pts2d, pts2d_projected):
    """ Compute residual error for each point.

    Parameters
    ----------
        pts2d (numpy.array): observed 2D (u, v) points of shape (N, 2). Where N is the number of points.
        pts2d_projected (numpy.array): 3D global points projected to 2D of shape (N, 2).
                                       Where N is the number of points.

    Returns
    -------
        residuals (numpy.array): residual error for each point (L2 distance between.
                                 each observed and projected 2D points). The array shape must be (N, 1).
                                 Where N is the same as in pts2d and pts2d_projected.
    """

    # TODO: Your code here
    pass  # TODO: Change to return residuals


def calibrate_camera(pts3d, pts2d, set_size_k):
    """ Find the best camera projection matrix given corresponding 3D and 2D points.

    Parameters
    ----------
        pts3d (numpy.array): 3D global (x, y, z) points of shape (N, 3). Where N is the number of points.
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.
        set_size_k (int): set of k random points to choose from pts2d.

    Returns
    -------
        bestM (numpy.array): best transformation matrix M of shape (3, 4).
        error (float): sum of squared residuals of all points for bestM.
        avg_residuals (numpy.array): Average residuals array, one row for each iteration.
                                     The array should be of shape (10, 1).
    """

    # TODO: Your code here
    pass  # TODO: Change to return bestM, error, avg_residuals


def get_camera_center(M):
    """ Find the camera global coordinates.

    Parameters
    ----------
        M (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).

    Returns
    -------
        center (numpy.array): [x, y, z] camera coordinates. Array must be of shape (1, 3).
    """

    # TODO: Your code here
    pass  # TODO: Change to return center


def compute_fundamental_matrix(pts2d_a, pts2d_b):
    """ Compute fundamental matrix given corresponding points from 2 images of a scene. Use the
    least-squares method, see np.linalg.lstsq

    Parameters
    ----------
        pts2d_a (numpy.array): 2D points from image A of shape (N, 2). Where N is the number of points.
        pts2d_b (numpy.array): 2D points from image B of shape (N, 2). Where N is the number of points.

    Returns
    -------
        F (numpy.array): fundamental matrix. Must be a (3, 3) array.
    """

    # TODO: Your code here
    pass  # TODO: Change to return F


def reduce_rank(F):
    """ Reduce a full rank (3, 3) matrix to rank 2.

    Parameters
    ----------
        F (numpy.array) : full rank fundamental matrix. Must be a (3, 3) array.

    Returns
    -------
        new_F (numpy.array) : rank 2 fundamental matrix. Must be a (3, 3) array.
    """

    # TODO: Your code here
    pass    # TODO: Change to return new_F


def get_epipolar_lines(a_shape, b_shape, F, pts2d_a, pts2d_b):
    """ Get epipolar lines using the fundamental matrix and two sets of 2D points.

    Parameters
    ----------
        a_shape (tuple): image A shape (rows, cols)
        b_shape (tuple): image B shape (rows, cols)
        F (numpy.array): Fundamental matrix of shape (3, 3).
        pts2d_a (numpy.array): 2D points from image A of shape (N, 2). Where N is the number of points.
        pts2d_b (numpy.array): 2D points from image B of shape (N, 2). Where N is the number of points.

    Returns
    -------
        epipolar_lines_a (list): epipolar lines for image A. Each list element should be [(x1_a, y1_a), (x2_a, y2_a)]
                                 one for each point in pts2d_a.
        epipolar_lines_b (list): epipolar lines for image B. Each list element should be [(x1_b, y1_b), (x2_b, y2_b)]
                                 one for each point in pts2d_b.
    """

    # TODO: Your code here
    pass    # TODO: Change to return epipolar_lines_a, epipolar_lines_b


def compute_T_matrix(pts2d):
    """ Compute the transformation matrix T given corresponding 2D points from an image.

    Parameters
    ----------
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.

    Returns
    -------
        T (numpy.array): transformation matrix T of shape (3, 3).
    """

    # TODO: Your code here
    pass    # TODO: Change to return T


def normalize_points(pts2d, T):
    """ Normalize 2D points.

    Parameters
    ----------
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.
        T (numpy.array): transformation matrix T of shape (3, 3).

    Returns
    -------
        pts2d_norm (numpy.array): normalized points (N, 2) array.
    """

    # TODO: Your code here
    pass    # TODO: Change to return pts2d_norm
