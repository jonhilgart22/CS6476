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
    # construct u as 11x1 matrix
    x = np.insert(pts3d, pts3d.shape[1], [1], axis=1)
    x1 = np.insert(x, x.shape[1], [[0], [0], [0], [0]], axis=1)
    x2 = pts3d * -1 * pts2d[:, 0].reshape(pts2d.shape[0], 1)
    u = np.concatenate((x1, x2), axis=1)

    # construct v as 11x1 matrix
    y = np.insert(pts3d, pts3d.shape[1], [1], axis=1)
    y1 = np.insert(y, 0, [[0], [0], [0], [0]], axis=1)
    y2 = pts3d * -1 * pts2d[:, 1].reshape(pts2d.shape[0], 1)
    v = np.concatenate((y1, y2), axis=1)

    # combined u and v together
    a = np.concatenate((u, v))
    # shape b so it matches the order of a
    b = pts2d.reshape((pts2d.size, 1), order='F')

    # calculate for M and error
    M, residual = np.linalg.lstsq(a, b)[:2]
    M = np.concatenate((M, [[1]])).reshape(3, 4)
    error = np.sum(np.square(residual))

    return M, error


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
    # add 4th dimension
    pts3d_temp = np.insert(pts3d, pts3d.shape[1], [1], axis=1)

    # dot product of matrix M with pts3d
    pts2d_projected = np.dot(M, pts3d_temp.T)
    # normalize (first two rows divided by the last column)
    pts2d_projected = pts2d_projected[:2] / pts2d_projected[2]
    # switch axis
    pts2d_projected = np.swapaxes(pts2d_projected, 0, 1)

    return pts2d_projected


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
    return np.linalg.norm(pts2d - pts2d_projected, axis=1).reshape(pts2d.shape[0], 1)


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
    lowest_residual = None
    bestM = None
    best_error = None

    for i in xrange(10):
        # Randomly choose k points from the 2D list and their corresponding points in the 3D list.
        indices = np.sort(np.random.choice(pts2d.shape[0], set_size_k, replace=False))

        # Compute the projection matrix M on the chosen points.
        M, error = solve_least_squares(pts3d[indices], pts2d[indices])

        # set probability to the chosen indices to 0 and split the rest
        p = [0 if i in indices else (1. / (pts2d.shape[0] - set_size_k)) for i in xrange(pts2d.shape[0])]

        # Pick 4 points not in your set of k, and compute the average residual.
        residual_indices = np.sort(np.random.choice(pts2d.shape[0], 4, replace=False, p=p))

        # get the projected 2d points of the 4 points
        pts2d_projected = project_points(pts3d[residual_indices], M)

        # get the residuals of the 4 points
        residuals = get_residuals(pts2d[residual_indices], pts2d_projected)

        # get the average residual
        avg_residual = np.average(residuals)

        # choose the lowest average residual
        if lowest_residual is None or lowest_residual > avg_residual:
            lowest_residual = avg_residual
            bestM = M
            best_error = error

    return bestM, best_error, lowest_residual


def get_camera_center(M):
    """ Find the camera global coordinates.

    Parameters
    ----------
        M (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).

    Returns
    -------
        center (numpy.array): [x, y, z] camera coordinates. Array must be of shape (1, 3).
    """
    # M = [Q|m4]
    # Center = -inv(Q) * m4
    center = np.sum(-1 * np.linalg.inv(M[:, :3]) * M[:, 3], axis=1).reshape(1, 3)
    return center


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
    # create [u, v, 1]
    uv1 = np.insert(pts2d_a, pts2d_a.shape[1], [1], axis=1)
    # create [u' * u, u' * v, u']
    a1 = uv1 * pts2d_b[:, :1]
    # create [v' * u, v' * v, v']
    a2 = uv1 * pts2d_b[:, 1:]
    # merge to [u' * u, u' * v, u', v' * u, v' * v, v', u, v]
    a = np.concatenate((a1, a2, uv1[:, :-1]), axis=1)
    # move the last element in a to the right side; b is [[-1],...N]
    b = -1 * uv1[:, :-2:-1]
    F, = np.linalg.lstsq(a, b)[:1]
    # add F33=1
    F = np.concatenate((F, [[1]])).reshape(3, 3)
    return F


def reduce_rank(F):
    """ Reduce a full rank (3, 3) matrix to rank 2.

    Parameters
    ----------
        F (numpy.array) : full rank fundamental matrix. Must be a (3, 3) array.

    Returns
    -------
        new_F (numpy.array) : rank 2 fundamental matrix. Must be a (3, 3) array.
    """
    # Single Value Decomposition
    U, S, Vh = np.linalg.svd(F)
    # copy S
    S_prime = np.copy(S)
    # set the smallest singular value to 0
    S_prime[2] = 0
    # recompute F
    new_F = np.dot(U, np.dot(np.diag(S_prime), Vh))
    return new_F


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
    # make homogeneous
    p_a = np.insert(pts2d_a, pts2d_a.shape[1], [1], axis=1)
    p_b = np.insert(pts2d_b, pts2d_b.shape[1], [1], axis=1)

    # multiply F to each points
    l_a = np.dot(F.T, p_b.T).T
    l_b = np.dot(F, p_a.T).T

    # construct image boundaries
    p_ul = np.array([0, 0, 1])
    p_bl = np.array([0, b_shape[0] - 1, 1])
    p_ur = np.array([b_shape[1] - 1, 0, 1])
    p_br = np.array([b_shape[1] - 1, b_shape[0] - 1, 1])
    # get the line in b space
    l_L = np.cross(p_ul, p_bl)
    l_R = np.cross(p_ur, p_br)
    # get the corresponding points in b space
    lines_b_left = np.cross(l_b, l_L)
    lines_b_left = (lines_b_left[:, :2] / lines_b_left[:, :-2:-1]).astype(np.int)
    lines_b_right = np.cross(l_b, l_R)
    lines_b_right = (lines_b_right[:, :2] / lines_b_right[:, :-2:-1]).astype(np.int)

    # construct image boundaries
    p_ul = np.array([0, 0, 1])
    p_bl = np.array([0, a_shape[0] - 1, 1])
    p_ur = np.array([a_shape[1] - 1, 0, 1])
    p_br = np.array([a_shape[1] - 1, a_shape[0] - 1, 1])
    # get the line in a space
    l_L = np.cross(p_ul, p_bl)
    l_R = np.cross(p_ur, p_br)
    # get the corresponding points in a space
    lines_a_left = np.cross(l_a, l_L)
    lines_a_left = (lines_a_left[:, :2] / lines_a_left[:, :-2:-1]).astype(np.int)
    lines_a_right = np.cross(l_a, l_R)
    lines_a_right = (lines_a_right[:, :2] / lines_a_right[:, :-2:-1]).astype(np.int)

    # merge left and right points
    epipolar_lines_a = np.concatenate((lines_b_left, lines_b_right), axis=1).reshape(lines_b_left.shape[0], 2, 2)
    epipolar_lines_a = [[tuple(points[i]) for i in xrange(len(points))] for points in epipolar_lines_a]
    epipolar_lines_b = np.concatenate((lines_a_left, lines_a_right), axis=1).reshape(lines_a_left.shape[0], 2, 2)
    epipolar_lines_b = [[tuple(points[i]) for i in xrange(len(points))] for points in epipolar_lines_b]

    return epipolar_lines_a, epipolar_lines_b


def compute_T_matrix(pts2d):
    """ Compute the transformation matrix T given corresponding 2D points from an image.

    Parameters
    ----------
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.

    Returns
    -------
        T (numpy.array): transformation matrix T of shape (3, 3).
    """
    # get the mean of x and y
    c_u, c_v = np.mean(pts2d, axis=0)
    # estimate the std dev after subtracting the mean
    s = np.sqrt(np.sum(np.square(pts2d - c_v))/pts2d.size)
    # construct the scale as diagonal 3x3 matrix
    scale = np.diag([s, s, 1])
    # construct the offset
    offset = np.array([
        [1, 0, -c_u],
        [0, 1, -c_v],
        [0, 0,    1]
    ])
    # calculate T
    T = np.dot(scale, offset)
    return T


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
    if type(pts2d) != np.ndarray:
        return None
    # make pts2d homogeneous
    pts = np.insert(pts2d, pts2d.shape[1], [1], axis=1)
    # calculate the normalized points
    pts2d_norm = np.dot(T, pts.T).T[:, :2]
    return pts2d_norm
