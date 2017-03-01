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
    m, residual = np.linalg.lstsq(a, b)[:2]
    m = np.concatenate((m, [[1]])).reshape(3, 4)
    error = np.sum(np.square(residual))

    return m, error


def project_points(pts3d, m):
    """Projects each 3D point to 2D using the matrix M.

    Args:
        pts3d (numpy.array): 3D global (x, y, z) points of shape (N, 3). Where N is the number of points.
        m (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).

    Returns:
        numpy.array: projected 2D (u, v) points of shape (N, 2). Where N is the same as pts3d.
    """
    # add 4th dimension
    pts3d_temp = np.insert(pts3d, pts3d.shape[1], [1], axis=1)

    # dot product of matrix M with pts3d
    pts2d_projected = np.dot(m, pts3d_temp.T)
    # normalize (first two rows divided by the last column)
    pts2d_projected = pts2d_projected[:2] / pts2d_projected[2]
    # switch axis
    pts2d_projected = np.swapaxes(pts2d_projected, 0, 1)

    return pts2d_projected


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
    return np.linalg.norm(pts2d - pts2d_projected, axis=1).reshape(pts2d.shape[0], 1)


def calibrate_camera(pts3d, pts2d, set_size_k):
    """Finds the best camera projection matrix given corresponding 3D and 2D points.

    Args:
        pts3d (numpy.array): 3D global (x, y, z) points of shape (N, 3). Where N is the number of points.
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.
        set_size_k (int): set of k random points to choose from pts2d.

    Returns:
        tuple: three-element tuple containing:
               best_m (numpy.array): best transformation matrix M of shape (3, 4).
               error (float): sum of squared residuals of all points for best_m.
               avg_residuals (numpy.array): Average residuals array, one row for each iteration.
                                            The array should be of shape (10, 1).
    """
    lowest_residual = None
    best_m = None
    best_error = None
    avg_residuals = np.zeros((0, 1))

    for i in xrange(10):
        # Randomly choose k points from the 2D list and their corresponding points in the 3D list.
        indices = np.sort(np.random.choice(pts2d.shape[0], set_size_k, replace=False))

        # Compute the projection matrix M on the chosen points.
        m, error = solve_least_squares(pts3d[indices], pts2d[indices])

        # set probability to the chosen indices to 0 and split the rest
        p = [0 if i in indices else (1. / (pts2d.shape[0] - set_size_k)) for i in xrange(pts2d.shape[0])]

        # Pick 4 points not in your set of k, and compute the average residual.
        residual_indices = np.sort(np.random.choice(pts2d.shape[0], 4, replace=False, p=p))

        # get the projected 2d points of the 4 points
        pts2d_projected = project_points(pts3d[residual_indices], m)

        # get the residuals of the 4 points
        residuals = get_residuals(pts2d[residual_indices], pts2d_projected)

        # get the average residual
        avg_residual = np.average(residuals)
        avg_residuals = np.append(avg_residuals, [[avg_residual]], axis=0)

        # choose the lowest average residual
        if lowest_residual is None or lowest_residual > avg_residual:
            lowest_residual = avg_residual
            best_m = m
            best_error = error

    return best_m, best_error, avg_residuals


def get_camera_center(m):
    """Finds the camera global coordinates.

    Args:
        m (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).

    Returns:
        numpy.array: [x, y, z] camera coordinates. Array must be of shape (1, 3).
    """
    center = np.sum(-1 * np.linalg.inv(m[:, :3]) * m[:, 3], axis=1).reshape(1, 3)
    return center


def compute_fundamental_matrix(pts2d_1, pts2d_2):
    """Computes the fundamental matrix given corresponding points from 2 images of a scene.

    This function uses the least-squares method, see numpy.linalg.lstsq.

    Args:
        pts2d_1 (numpy.array): 2D points from image 1 of shape (N, 2). Where N is the number of points.
        pts2d_2 (numpy.array): 2D points from image 2 of shape (N, 2). Where N is the number of points.

    Returns:
        numpy.array: array containing the fundamental matrix elements. Array must be of shape (3, 3).
    """
    # create [u, v, 1]
    uv1 = np.insert(pts2d_1, pts2d_1.shape[1], [1], axis=1)
    # create [u' * u, u' * v, u']
    a1 = uv1 * pts2d_2[:, :1]
    # create [v' * u, v' * v, v']
    a2 = uv1 * pts2d_2[:, 1:]
    # merge to [u' * u, u' * v, u', v' * u, v' * v, v', u, v]
    a = np.concatenate((a1, a2, uv1[:, :-1]), axis=1)
    # move the last element in a to the right side; b is [[-1],...N]
    b = -1 * uv1[:, :-2:-1]
    f, = np.linalg.lstsq(a, b)[:1]
    # add F33=1
    f = np.concatenate((f, [[1]])).reshape(3, 3)
    return f


def reduce_rank(f):
    """Reduces a full rank (3, 3) matrix to rank 2.

    Args:
        f (numpy.array): full rank fundamental matrix. Must be a (3, 3) array.

    Returns:
        numpy.array: rank 2 fundamental matrix. Must be a (3, 3) array.
    """
    # Single Value Decomposition
    u, s, v_h = np.linalg.svd(f)
    # copy s
    s_prime = s.copy()
    # set the smallest singular value to 0
    s_prime[2] = 0
    # recompute F
    new_f = np.dot(u, np.dot(np.diag(s_prime), v_h))
    return new_f


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
    # make homogeneous
    p_a = np.insert(pts2d_1, pts2d_1.shape[1], [1], axis=1)
    p_b = np.insert(pts2d_2, pts2d_2.shape[1], [1], axis=1)

    # multiply F to each points
    l_a = np.dot(f.T, p_b.T).T
    l_b = np.dot(f, p_a.T).T

    # construct image boundaries
    p_ul = np.array([0, 0, 1])
    p_bl = np.array([0, img2_shape[0] - 1, 1])
    p_ur = np.array([img2_shape[1] - 1, 0, 1])
    p_br = np.array([img2_shape[1] - 1, img2_shape[0] - 1, 1])
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
    p_bl = np.array([0, img1_shape[0] - 1, 1])
    p_ur = np.array([img1_shape[1] - 1, 0, 1])
    p_br = np.array([img1_shape[1] - 1, img1_shape[0] - 1, 1])
    # get the line in a space
    l_L = np.cross(p_ul, p_bl)
    l_R = np.cross(p_ur, p_br)
    # get the corresponding points in a space
    lines_a_left = np.cross(l_a, l_L)
    lines_a_left = (lines_a_left[:, :2] / lines_a_left[:, :-2:-1]).astype(np.int)
    lines_a_right = np.cross(l_a, l_R)
    lines_a_right = (lines_a_right[:, :2] / lines_a_right[:, :-2:-1]).astype(np.int)

    # merge left and right points
    epipolar_lines_a = np.concatenate((lines_a_left, lines_a_right), axis=1).reshape(lines_a_left.shape[0], 2, 2)
    epipolar_lines_a = [[tuple(points[i]) for i in xrange(len(points))] for points in epipolar_lines_a]
    epipolar_lines_b = np.concatenate((lines_b_left, lines_b_right), axis=1).reshape(lines_b_left.shape[0], 2, 2)
    epipolar_lines_b = [[tuple(points[i]) for i in xrange(len(points))] for points in epipolar_lines_b]

    return epipolar_lines_a, epipolar_lines_b


def compute_t_matrix(pts2d):
    """Computes the transformation matrix T given corresponding 2D points from an image.

    Args:
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.

    Returns:
        numpy.array: transformation matrix T of shape (3, 3).
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
    t = np.dot(scale, offset)
    return t


def normalize_points(pts2d, t):
    """Normalizes 2D points.

    Args:
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.
        t (numpy.array): transformation matrix T of shape (3, 3).

    Returns:
        numpy.array: normalized points (N, 2) array.
    """
    if type(pts2d) != np.ndarray:
        return None
    # make pts2d homogeneous
    pts = np.insert(pts2d, pts2d.shape[1], [1], axis=1)
    # calculate the normalized points
    pts2d_norm = np.dot(t, pts.T).T[:, :2]
    return pts2d_norm
