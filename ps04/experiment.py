"""Problem Set 4: Geometry."""

import numpy as np
import cv2
import os

from ps4 import *

# I/O directories
input_dir = "input"
output_dir = "output"

# Input files
PIC_A = "pic_a.jpg"
PIC_A_2D = "pts2d-pic_a.txt"
PIC_A_2D_NORM = "pts2d-norm-pic_a.txt"
PIC_B = "pic_b.jpg"
PIC_B_2D = "pts2d-pic_b.txt"
SCENE = "pts3d.txt"
SCENE_NORM = "pts3d-norm.txt"


# Utility code
def read_points(filename):
    """ Read point data from given file and return as NumPy array.

    Parameters
    ----------
        filename (string): name of the file to open.

    Returns
    -------
        pts (numpy.array): Array with the points data.
    """
    with open(filename) as f:
        lines = f.readlines()
        pts = []
        for line in lines:
            pts.append(map(float, line.split()))
    return np.array(pts)


def draw_epipolar_lines(img_in, line_points, color=(255, 0, 0)):
    """ Draw epipolar lines.

    Parameters
    ----------
        img_in (numpy.array): input image.
        line_points (list): List of tuples where each element consists of [(x1, y1), (x2, y2)].
        color (triple): Lines color.

    Returns
    -------
        img_out (numpy.array): image with epipolar lines drawn.
    """
    img_out = np.copy(img_in)

    for points in line_points:
        cv2.line(img_out, (points[0]), (points[1]), color)

    return img_out


def main():
    """
    Driver code. Feel free to modify this code as this is to help you get the
    answers for your report.
    """

    # 2. AVERAGE FOR BEST_M
    # NOTE: These points are not normalized
    pts3d = read_points(os.path.join(input_dir, SCENE))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))

    best_M_8, error_8, avg_residuals_8 = 0, 0, 0
    best_M_12, error_12, avg_residuals_12 = 0, 0, 0
    best_M_16, error_16, avg_residuals_16 = 0, 0, 0
    num = 100
    for i in xrange(num):
        _best_M_8, _error_8, _avg_residuals_8 = calibrate_camera(pts3d, pts2d_pic_b, 8)  # TODO: implement this
        _best_M_12, _error_12, _avg_residuals_12 = calibrate_camera(pts3d, pts2d_pic_b, 12)  # TODO: implement this
        _best_M_16, _error_16, _avg_residuals_16 = calibrate_camera(pts3d, pts2d_pic_b, 16)  # TODO: implement this
        best_M_8 += _best_M_8
        best_M_12 += _best_M_12
        best_M_16 += _best_M_16
        error_8 += _error_8
        error_12 += _error_12
        error_16 += _error_16
        avg_residuals_8 += _avg_residuals_8
        avg_residuals_12 += _avg_residuals_12
        avg_residuals_16 += _avg_residuals_16

    best_M_8 /= num
    best_M_12 /= num
    best_M_16 /= num
    error_8 /= num
    error_12 /= num
    error_16 /= num
    avg_residuals_8 /= num
    avg_residuals_12 /= num
    avg_residuals_16 /= num

    print '\n Best M k = 8: \n', best_M_8
    print '\n Error Best M k = 8: \n', error_8
    print ' Average Residuals k = 8: \n', avg_residuals_8

    print '\n Best M k = 12: \n', best_M_12
    print '\n Error Best M k = 12: \n', error_12
    print ' Average Residuals k = 12: \n', avg_residuals_12

    print '\n Best M k = 16: \n', best_M_16
    print '\n Error Best M k = 16: \n', error_16
    print ' Average Residuals k = 16: \n', avg_residuals_16

    # 1
    # Read points
    pts3d_norm = read_points(os.path.join(input_dir, SCENE_NORM))
    pts2d_norm_pic_a = read_points(os.path.join(input_dir, PIC_A_2D_NORM))

    # Solve for transformation matrix using least squares
    M, error = solve_least_squares(pts3d_norm, pts2d_norm_pic_a)  # TODO: implement this
    print '\n M matrix using normalized points: \n', M
    print '\n Error value using normalized points: \n', error

    # Project 3D points to 2D
    pts2d_projected = project_points(pts3d_norm, M)  # TODO: implement this
    print '\n Projection of the last normalized point: \n', pts2d_projected[-1]

    # Compute residual error for each point
    residuals = get_residuals(pts2d_norm_pic_a, pts2d_projected)  # TODO: implement this
    print '\n Residuals: \n', residuals

    # 2
    # NOTE: These points are not normalized
    pts3d = read_points(os.path.join(input_dir, SCENE))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))

    best_M_8, error_8, avg_residuals_8 = calibrate_camera(pts3d, pts2d_pic_b, 8)  # TODO: implement this
    print '\n Best M k = 8: \n', best_M_8
    print '\n Error Best M k = 8: \n', error_8
    print '\n Average Residuals k = 8: \n', avg_residuals_8

    best_M_12, error_12, avg_residuals_12 = calibrate_camera(pts3d, pts2d_pic_b, 12)  # TODO: implement this
    print '\n Best M k = 12: \n', best_M_12
    print '\n Error Best M k = 12: \n', error_12
    print '\n Average Residuals k = 12: \n', avg_residuals_12

    best_M_16, error_16, avg_residuals_16 = calibrate_camera(pts3d, pts2d_pic_b, 16)  # TODO: implement this
    print '\n Best M k = 16: \n', best_M_16
    print '\n Error Best M k = 16: \n', error_16
    print '\n Average Residuals k = 16: \n', avg_residuals_16

    # Todo: Pick the best M from k = 8, 12, or 16
    if avg_residuals_8 <= avg_residuals_12:
        best_M = best_M_8
        best_avg_residuals = avg_residuals_8
    else:
        best_M = best_M_12
        best_avg_residuals = avg_residuals_12

    if best_avg_residuals > avg_residuals_16:
        best_M = best_M_16

    camera_center = get_camera_center(best_M)   # TODO: implement this
    print '\n Camera center: \n', camera_center

    # 3
    pts2d_pic_a = read_points(os.path.join(input_dir, PIC_A_2D))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))

    F_tilde = compute_fundamental_matrix(pts2d_pic_a, pts2d_pic_b)    # TODO: implement this
    print '\n Fundamental Matrix F', 'rank = ', np.linalg.matrix_rank(F_tilde), '\n', F_tilde

    F = reduce_rank(F_tilde)    # TODO: implement this
    print '\n Fundamental Matrix F', 'rank = ', np.linalg.matrix_rank(F), '\n', F

    img_a = cv2.imread(os.path.join(input_dir, PIC_A))
    img_b = cv2.imread(os.path.join(input_dir, PIC_B))

    # TODO: implement get_epipolar_lines
    lines_img_a, lines_img_b = get_epipolar_lines(img_a.shape, img_b.shape, F, pts2d_pic_a, pts2d_pic_b)

    epi_img_a = draw_epipolar_lines(img_b, lines_img_a)  # Implemented in the utility code
    epi_img_b = draw_epipolar_lines(img_a, lines_img_b)  # Implemented in the utility code

    cv2.imwrite(os.path.join(output_dir, 'ps4-2-c-1.png'), epi_img_a)
    cv2.imwrite(os.path.join(output_dir, 'ps4-2-c-2.png'), epi_img_b)

    # 4
    T_a = compute_T_matrix(pts2d_pic_a)  # TODO: implement this
    T_b = compute_T_matrix(pts2d_pic_b)  # TODO: implement this

    pts2d_pic_a_norm = normalize_points(pts2d_pic_a, T_a)  # TODO: implement this
    pts2d_pic_b_norm = normalize_points(pts2d_pic_b, T_b)  # TODO: implement this

    F_hat = compute_fundamental_matrix(pts2d_pic_a_norm, pts2d_pic_b_norm)
    F_hat = reduce_rank(F_hat)

    print '\n T_a matrix: \n', T_a
    print '\n T_b matrix: \n', T_b
    print '\n Fundamental Matrix F: \n', F_hat

    new_F = np.dot(T_b.T, np.dot(F_hat, T_a))  # TODO: insert the code for the matrix multiplication of transpose(T_b), F, T_a)
    print '\n New Fundamental Matrix F: \n', new_F

    lines_img_a, lines_img_b = get_epipolar_lines(img_a.shape, img_b.shape, new_F, pts2d_pic_a, pts2d_pic_b)

    epi_img_a = draw_epipolar_lines(img_b, lines_img_a)
    epi_img_b = draw_epipolar_lines(img_a, lines_img_b)

    cv2.imwrite(os.path.join(output_dir, 'ps4-2-e-1.png'), epi_img_a)
    cv2.imwrite(os.path.join(output_dir, 'ps4-2-e-2.png'), epi_img_b)

if __name__ == '__main__':
    main()
