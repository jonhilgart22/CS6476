"""Problem Set 5: Harris, ORB, RANSAC"""

import numpy as np
import cv2
import os

from ps5 import *

input_dir = "input"
output_dir = "output"


# Utility code
def find_and_draw_corners(image, harris_resp, threshold, radius, output_filename):
    corners = find_corners(harris_resp, threshold=threshold, radius=radius)  # TODO: implement this
    image_out = draw_corners(image, corners)  # TODO: implement this

    image_out = cv2.normalize(image_out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, output_filename), image_out)

    return corners


def main():
    """
    Driver code. Feel free to modify this code as this is to help you get the
    answers for your report.
    """

    # check = cv2.imread(os.path.join(input_dir, 'test_out1.png'), 0) / 255.
    # check_x = gradient_x(check)
    # check_y = gradient_y(check)
    # check_pair = make_image_pair(check_x, check_y)
    # check_pair = cv2.normalize(check_pair, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "check_pair.png"), check_pair)
    #
    # kernel_dims = (3, 3)
    # alpha = 0.06
    # check_r = harris_response(check_x, check_y, kernel_dims, alpha)
    # check_r = cv2.normalize(check_r, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "check_r.png"), check_r)
    #
    # threshold = 0.6
    # radius = 3
    # check_corners = find_corners(check_r, threshold, radius)
    # check_out = draw_corners(check, check_corners)
    # check_out = cv2.normalize(check_out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "check_out.png"), check_out)
    #
    # check_rot = cv2.imread(os.path.join(input_dir, 'test_out2.png'), 0) / 255.
    # check_rot_x = gradient_x(check_rot)
    # check_rot_y = gradient_y(check_rot)
    # check_rot_pair = make_image_pair(check_rot_x, check_rot_y)
    # check_rot_pair = cv2.normalize(check_rot_pair, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "check_rot_pair.png"), check_rot_pair)
    #
    # check_rot_r = harris_response(check_rot_x, check_rot_y, kernel_dims, alpha)
    # check_rot_r = cv2.normalize(check_rot_r, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "check_rot_r.png"), check_rot_r)
    #
    # check_rot_corners = find_corners(check_rot_r, threshold, radius)
    # check_rot_out = draw_corners(check_rot, check_rot_corners)
    # check_rot_out = cv2.normalize(check_rot_out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(os.path.join(output_dir, "check_rot_out.png"), check_rot_out)



    # 1a
    trans_a = cv2.imread(os.path.join(input_dir, "transA.jpg"), 0) / 255.
    trans_a_x = gradient_x(trans_a)  # TODO: implement this
    trans_a_y = gradient_y(trans_a)  # TODO: implement this
    trans_a_pair = make_image_pair(trans_a_x, trans_a_y)  # TODO: implement this

    trans_a_pair = cv2.normalize(trans_a_pair, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-a-1.png"), trans_a_pair)

    sim_a = cv2.imread(os.path.join(input_dir, "simA.jpg"), 0) / 255.
    sim_a_x = gradient_x(sim_a)  # TODO: implement this
    sim_a_y = gradient_y(sim_a)  # TODO: implement this
    sim_a_pair = make_image_pair(sim_a_x, sim_a_y)  # TODO: implement this

    sim_a_pair = cv2.normalize(sim_a_pair, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-a-2.png"), sim_a_pair)

    # Load the remaining images
    trans_b = cv2.imread(os.path.join(input_dir, "transB.jpg"), 0) / 255.
    sim_b = cv2.imread(os.path.join(input_dir, "simB.jpg"), 0) / 255.

    # 1b
    kernel_dims = (3, 3)  # TODO: Define a the kernel dimensions (tuple)
    alpha = 0.06  # TODO: Set a float value for alpha
    trans_a_r = harris_response(trans_a_x, trans_a_y, kernel_dims, alpha)  # TODO: implement this
    trans_a_r = cv2.normalize(trans_a_r, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-b-1.png"), trans_a_r)

    trans_b_x = gradient_x(trans_b)
    trans_b_y = gradient_y(trans_b)
    trans_b_r = harris_response(trans_b_x, trans_b_y, kernel_dims, alpha)  # you can use a different kernel_dims and alpha
    trans_b_r = cv2.normalize(trans_b_r, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-b-2.png"), trans_b_r)

    sim_a_r = harris_response(sim_a_x, sim_a_y, kernel_dims, alpha)  # you can use a different kernel_dims and alpha
    sim_a_r = cv2.normalize(sim_a_r, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-b-3.png"), sim_a_r)

    sim_b_x = gradient_x(sim_b)
    sim_b_y = gradient_y(sim_b)
    sim_b_r = harris_response(sim_b_x, sim_b_y, kernel_dims, alpha)  # you can use a different kernel_dims and alpha
    sim_b_r = cv2.normalize(sim_b_r, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-b-4.png"), sim_b_r)

    # 1c
    threshold = 0.5  # TODO: define a threshold value
    radius = 5  # TODO: define a radius value
    trans_a_corners = find_corners(trans_a_r, threshold, radius)  # TODO: implement this
    trans_a_out = draw_corners(trans_a, trans_a_corners)  # TODO: implement this
    trans_a_out = cv2.normalize(trans_a_out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "ps5-1-c-1.png"), trans_a_out)

    trans_b_corners = find_and_draw_corners(trans_b, trans_b_r, threshold, radius,  # you can use a different threshold and radius
                                            output_filename="ps5-1-c-2.png")

    sim_a_corners = find_and_draw_corners(sim_a, sim_a_r, threshold, radius,  # you can use a different threshold and radius
                                          output_filename="ps5-1-c-3.png")

    sim_b_corners = find_and_draw_corners(sim_b, sim_b_r, threshold, radius,  # you can use a different threshold and radius
                                          output_filename="ps5-1-c-4.png")

    # 2a
    size = 10  # TODO: Define a size value
    octave = 0  # You can leave this value to 0
    trans_a_angle = gradient_angle(trans_a_x, trans_a_y)  # TODO: implement this
    trans_a_kp, trans_a_out = get_draw_keypoints(trans_a_corners, trans_a_r, trans_a_angle, size, octave)   # TODO: implement this
    trans_b_angle = gradient_angle(trans_b_x, trans_b_y)  # TODO: implement this
    trans_b_kp, trans_b_out = get_draw_keypoints(trans_b_corners, trans_b_r, trans_b_angle, size, octave)

    trans_a_b_out = make_image_pair(trans_a_out, trans_b_out)
    cv2.imwrite(os.path.join(output_dir, "ps5-2-a-1.png"), trans_a_b_out)

    sim_a_angle = gradient_angle(sim_a_x, sim_a_y)  # TODO: implement this
    sim_a_kp, sim_a_out = get_draw_keypoints(sim_a_corners, sim_a_r, sim_a_angle, size, octave)  # TODO: implement this
    sim_b_angle = gradient_angle(sim_b_x, sim_b_y)  # TODO: implement this
    sim_b_kp, sim_b_out = get_draw_keypoints(sim_b_corners, sim_b_r, sim_b_angle, size, octave)

    sim_a_b_out = make_image_pair(sim_a_out, sim_b_out)
    cv2.imwrite(os.path.join(output_dir, "ps5-2-a-2.png"), sim_a_b_out)

    # 2b
    trans_a_des, trans_a_kp = get_descriptors(trans_a, trans_a_kp)
    trans_b_des, trans_b_kp = get_descriptors(trans_b, trans_b_kp)
    sim_a_des, sim_a_kp = get_descriptors(sim_a, sim_a_kp)
    sim_b_des, sim_b_kp = get_descriptors(sim_b, sim_b_kp)

    trans_matches = match_descriptors(trans_a_des, trans_b_des)  # TODO: implement this
    sim_matches = match_descriptors(sim_a_des, sim_b_des)

    trans_a_b_out = draw_matches(trans_a, trans_b, trans_a_kp, trans_b_kp, trans_matches)
    cv2.imwrite(os.path.join(output_dir, "ps5-2-b-1.png"), trans_a_b_out)

    sim_a_b_out = draw_matches(sim_a, sim_b, sim_a_kp, sim_b_kp, sim_matches)
    cv2.imwrite(os.path.join(output_dir, "ps5-2-b-2.png"), sim_a_b_out)

    # 3a
    threshold = 15  # TODO: define a threshold value
    translation, good_matches = compute_translation_RANSAC(trans_a_kp, trans_b_kp, trans_matches,
                                                           threshold)  # TODO: implement this
    # trans_pair = make_image_pair(trans_a, trans_b)
    print '3a: Translation vector: \n', translation

    # TODO: Draw biggest consensus set lines on trans_pair. Try to use different colors for each line.
    trans_pair = draw_matches(trans_a, trans_b, trans_a_kp, trans_b_kp, good_matches)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-a-1.png"), trans_pair)

    # 3b
    threshold = 20  # TODO: define a threshold value
    similarity, sim_good_matches = compute_similarity_RANSAC(sim_a_kp, sim_b_kp, sim_matches,
                                                             threshold)  # TODO: implement this
    print '3b: Transform Matrix for the best set: \n', similarity

    # TODO: Draw biggest consensus set lines on trans_pair. Try to use different colors for each line.
    sim_pair = draw_matches(sim_a, sim_b, sim_a_kp, sim_b_kp, sim_good_matches)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-b-1.png"), sim_pair)

    # 3c
    threshold = 20  # TODO: define a threshold value
    similarity_affine, sim_good_matches = compute_affine_RANSAC(sim_a_kp, sim_b_kp, sim_matches,
                                                                threshold)  # TODO: implement this
    print '3c: Transform Matrix for the best set: \n', similarity_affine

    # TODO: Draw biggest consensus set lines on trans_pair. Try to use different colors for each line.
    sim_pair = draw_matches(sim_a, sim_b, sim_a_kp, sim_b_kp, sim_good_matches)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-c-1.png"), sim_pair)

    # 3d
    warpedB, overlay = warp_img(sim_a, sim_b, similarity)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-d-1.png"), warpedB)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-d-2.png"), overlay)

    # 4a
    warpedB, overlay = warp_img(sim_a, sim_b, similarity_affine)
    cv2.imwrite(os.path.join(output_dir, "ps5-4-a-1.png"), warpedB)
    cv2.imwrite(os.path.join(output_dir, "ps5-4-a-2.png"), overlay)

if __name__ == '__main__':
    main()
