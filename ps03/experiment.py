"""Problem Set 3: Geometry."""

import numpy as np
import cv2
import os

from ps3 import *

input_dir = "input"
output_dir = "output"


def main():
    """
    Driver code. Feel free to modify this code as this is to help you get the
    answers for your report.
    """

    # 1 a
    L = cv2.imread(os.path.join(input_dir, 'pair0-L.png'), 0) / 255.  # grayscale, scale to [0.0, 1.0]
    R = cv2.imread(os.path.join(input_dir, 'pair0-R.png'), 0) / 255.

    # Compute disparity using method disparity_ssd.
    # TODO: You need to define the window size and the maximum value of disparity
    w_size = 5  # TODO: define the window size (int)
    dmax = 5  # TODO: define the max disparity (int)
    D_L = disparity_ssd(L, R, 1, w_size, dmax)
    D_R = disparity_ssd(R, L, 0, w_size, dmax)

    # TODO: Adjust the image signs to match the color convention you plan to use
    D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite(os.path.join(output_dir, 'ps3-1-a-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-1-a-2.png'), D_R)

    # 1 b
    L = cv2.imread(os.path.join(input_dir, 'pair1-L.png'), 0) / 255.
    R = cv2.imread(os.path.join(input_dir, 'pair1-R.png'), 0) / 255.

    w_size = 7  # TODO: define the window size (int)
    dmax = 100  # TODO: define the max disparity (int)
    D_L = disparity_ssd(L, R, 1, w_size, dmax)
    D_R = disparity_ssd(R, L, 0, w_size, dmax)

    # TODO: Adjust the image signs to match the color convention you plan to use
    D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite(os.path.join(output_dir, 'ps3-1-b-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-1-b-2.png'), D_R)

    # 2 a
    # TODO: Choose to either add noise to only one image or both
    sigma = 0.05  # TODO: choose a value for sigma
    L_noisy = add_noise(L, sigma)
    R_noisy = add_noise(R, sigma)

    cv2.imwrite(os.path.join(output_dir, 'pair1-L.png'), L * 255)
    cv2.imwrite(os.path.join(output_dir, 'pair1-R.png'), R * 255)
    cv2.imwrite(os.path.join(output_dir, 'pair1-L-noisy.png'), L_noisy * 255)
    cv2.imwrite(os.path.join(output_dir, 'pair1-R-noisy.png'), R_noisy * 255)

    # TODO: Change the following four lines based on whether you are using just one noisy image or both
    image_L = L_noisy  # TODO: can be L or L_noisy
    image_R = R_noisy  # TODO: can be R or R_noisy
    w_size = 7  # TODO: define the window size (int)
    dmax = 100  # TODO: define the max disparity (int)
    D_L = disparity_ssd(image_L, image_R, 1, w_size, dmax)
    D_R = disparity_ssd(image_R, image_L, 0, w_size, dmax)

    # TODO: Adjust the image signs to match the color convention you plan to use
    D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite(os.path.join(output_dir, 'ps3-2-a-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-2-a-2.png'), D_R)

    # 2 b
    # TODO: Choose an image to increase contrast to
    value = 10  # percent (%).
    image_to_boost = L  # Todo: image_to_boost can be either L or R
    contrast_img = increase_contrast(image_to_boost, value)

    cv2.imwrite(os.path.join(output_dir, 'contrast_img1.png'), L * 255)
    cv2.imwrite(os.path.join(output_dir, 'contrast_img2.png'), contrast_img * 255)

    # TODO: Change the following two lines accordingly
    image_L = contrast_img  # TODO: can be either L or contrast_img if L was used
    image_R = R  # TODO: can be either R or contrast_img if R was used
    w_size = 7  # TODO: define the window size (int)
    dmax = 120  # TODO: define the max disparity (int)
    D_L = disparity_ssd(image_L, image_R, 1, w_size, dmax)
    D_R = disparity_ssd(image_R, image_L, 0, w_size, dmax)

    # TODO: Adjust the image signs to match the color convention you plan to use
    D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite(os.path.join(output_dir, 'ps3-2-b-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-2-b-2.png'), D_R)

    # 3. Compute disparity using method disparity_ncorr.
    # 3 a
    # TODO: You need to define the window size and the maximum value of disparity
    w_size = 7  # TODO: define the window size (int)
    dmax = 100  # TODO: define the max disparity (int)
    D_L = disparity_ncorr(L, R, 1, w_size, dmax)
    D_R = disparity_ncorr(R, L, 0, w_size, dmax)

    # TODO: Adjust the image signs to match the color convention you plan to use
    D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite(os.path.join(output_dir, 'ps3-3-a-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-3-a-2.png'), D_R)

    # 3 b 1
    # TODO: Use the image(s) with added noise from 2a.
    # TODO: Change the following two lines based on whether you are using just one noisy image or both
    image_L = L_noisy
    image_R = R_noisy
    w_size = 7  # TODO: define the window size (int)
    dmax = 100  # TODO: define the max disparity (int)
    D_L = disparity_ncorr(image_L, image_R, 1, w_size, dmax)
    D_R = disparity_ncorr(image_R, image_L, 0, w_size, dmax)

    # TODO: Adjust the image signs to match the color convention you plan to use
    D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-2.png'), D_R)

    # 3 b 2
    # TODO: Use the image(s) with increased contrast from 2b.
    # TODO: Change the following two lines accordingly
    image_L = contrast_img
    image_R = R
    w_size = 7  # TODO: define the window size (int)
    dmax = 100  # TODO: define the max disparity (int)
    D_L = disparity_ncorr(image_L, image_R, 1, w_size, dmax)
    D_R = disparity_ncorr(image_R, image_L, 0, w_size, dmax)

    # TODO: Adjust the image signs to match the color convention you plan to use
    D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-3.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-4.png'), D_R)

    # 4
    L = cv2.imread(os.path.join(input_dir, 'pair2-L.png'), 0) / 255.
    R = cv2.imread(os.path.join(input_dir, 'pair2-R.png'), 0) / 255.

    # TODO: Try your algorithms on pair2. Play with the images: smooth,
    #       sharpen, etc. Try both your normalized correlation and SSD methods.
    #       Keep comparing your results to the ground truth (pair2-D_L.png and
    #       pair2-D_R.png)

    filter_size = 5
    L_smoothed = cv2.GaussianBlur(L, (filter_size, filter_size), sigmaX=0, sigmaY=0)
    R_smoothed = cv2.GaussianBlur(R, (filter_size, filter_size), sigmaX=0, sigmaY=0)

    image_L = increase_contrast(L_smoothed, 10)
    image_R = R_smoothed  # increase_contrast(R_smoothed, 50)

    w_size = 7  # TODO: define the window size (int)
    dmax = 100  # TODO: define the max disparity (int)
    D_L = disparity_ncorr(image_L, image_R, 1, w_size, dmax)
    D_R = disparity_ncorr(image_R, image_L, 0, w_size, dmax)

    D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite(os.path.join(output_dir, 'ps3-4-a-1.png'), D_L)
    cv2.imwrite(os.path.join(output_dir, 'ps3-4-a-2.png'), D_R)

    # w_size_list = [5, 6, 7, 8, 9, 10]
    # dmax_list = [90, 95, 100, 105, 110, 115, 120, 125, 130]
    # # w_size_list = [7, 10]
    # # dmax_list = [135, 140, 145, 150, 155, 160]
    # L = cv2.imread(os.path.join(input_dir, 'pair1-L.png'), 0) / 255.
    # R = cv2.imread(os.path.join(input_dir, 'pair1-R.png'), 0) / 255.
    #
    # sigma = 0.05  # TODO: choose a value for sigma
    # L_noisy = add_noise(L, sigma)
    # R_noisy = add_noise(R, sigma)
    #
    # value = 10  # percent (%).
    # image_to_boost = L  # Todo: image_to_boost can be either L or R
    # contrast_img = increase_contrast(image_to_boost, value)
    #
    # for w_size in w_size_list:
    #     for dmax in dmax_list:
    #         data = {
    #             'w_size': w_size,
    #             'dmax': dmax
    #         }
    #
    #         # # 1 b
    #         # D_L = disparity_ssd(L, R, 1, w_size, dmax)
    #         # D_R = disparity_ssd(R, L, 0, w_size, dmax)
    #         #
    #         # # TODO: Adjust the image signs to match the color convention you plan to use
    #         # D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         # D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         #
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-1-b-1.%(w_size)s-%(dmax)s.png' % data), D_L)
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-1-b-2.%(w_size)s-%(dmax)s.png' % data), D_R)
    #
    #         # # 2 a
    #         # # TODO: Choose to either add noise to only one image or both
    #         # image_L = L_noisy  # TODO: can be L or L_noisy
    #         # image_R = R_noisy  # TODO: can be R or R_noisy
    #         # D_L = disparity_ssd(image_L, image_R, 1, w_size, dmax)
    #         # D_R = disparity_ssd(image_R, image_L, 0, w_size, dmax)
    #         #
    #         # # TODO: Adjust the image signs to match the color convention you plan to use
    #         # D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         # D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         #
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-2-a-1.%(w_size)s-%(dmax)s.png' % data), D_L)
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-2-a-2.%(w_size)s-%(dmax)s.png' % data), D_R)
    #
    #         # # 2 b
    #         # # TODO: Change the following two lines accordingly
    #         # image_L = contrast_img  # TODO: can be either L or contrast_img if L was used
    #         # image_R = R  # TODO: can be either R or contrast_img if R was used
    #         # D_L = disparity_ssd(image_L, image_R, 1, w_size, dmax)
    #         # D_R = disparity_ssd(image_R, image_L, 0, w_size, dmax)
    #         #
    #         # # TODO: Adjust the image signs to match the color convention you plan to use
    #         # D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         # D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         #
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-2-b-1.%(w_size)s-%(dmax)s.png' % data), D_L)
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-2-b-2.%(w_size)s-%(dmax)s.png' % data), D_R)
    #
    #         # # 3. Compute disparity using method disparity_ncorr.
    #         # # 3 a
    #         # # TODO: You need to define the window size and the maximum value of disparity
    #         # D_L = disparity_ncorr(L, R, 1, w_size, dmax)
    #         # D_R = disparity_ncorr(R, L, 0, w_size, dmax)
    #         #
    #         # # TODO: Adjust the image signs to match the color convention you plan to use
    #         # D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         # D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         #
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-3-a-1.%(w_size)s-%(dmax)s.png' % data), D_L)
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-3-a-2.%(w_size)s-%(dmax)s.png' % data), D_R)
    #
    #         # # 3 b 1
    #         # image_L = L_noisy
    #         # image_R = R_noisy
    #         # D_L = disparity_ncorr(image_L, image_R, 1, w_size, dmax)
    #         # D_R = disparity_ncorr(image_R, image_L, 0, w_size, dmax)
    #         #
    #         # # TODO: Adjust the image signs to match the color convention you plan to use
    #         # D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         # D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         #
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-1.%(w_size)s-%(dmax)s.png' % data), D_L)
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-2.%(w_size)s-%(dmax)s.png' % data), D_R)
    #
    #         # # 3 b 2
    #         # # TODO: Use the image(s) with increased contrast from 2b.
    #         # # TODO: Change the following two lines accordingly
    #         # image_L = contrast_img
    #         # image_R = R
    #         # D_L = disparity_ncorr(image_L, image_R, 1, w_size, dmax)
    #         # D_R = disparity_ncorr(image_R, image_L, 0, w_size, dmax)
    #         #
    #         # # TODO: Adjust the image signs to match the color convention you plan to use
    #         # D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         # D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         #
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-3.%(w_size)s-%(dmax)s.png' % data), D_L)
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-4.%(w_size)s-%(dmax)s.png' % data), D_R)
    #
    #         # # 4
    #         # # TODO: Try your algorithms on pair2. Play with the images: smooth,
    #         # #       sharpen, etc. Try both your normalized correlation and SSD methods.
    #         # #       Keep comparing your results to the ground truth (pair2-D_L.png and
    #         # #       pair2-D_R.png)
    #         #
    #         # image_L = increase_contrast(L_smoothed, 10)
    #         # image_R = R_smoothed  # increase_contrast(R_smoothed, 50)
    #         # D_L = disparity_ncorr(image_L, image_R, 1, w_size, dmax)
    #         # D_R = disparity_ncorr(image_R, image_L, 0, w_size, dmax)
    #         #
    #         # D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         # D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #         #
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-4-a-1.%(w_size)s-%(dmax)s.png' % data), D_L)
    #         # cv2.imwrite(os.path.join(output_dir, 'ps3-4-a-2.%(w_size)s-%(dmax)s.png' % data), D_R)

if __name__ == '__main__':
    main()
