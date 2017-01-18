"""Problem Set 2: Edges and Lines"""
import os
import math
from ps2 import *

# I/O directories
input_dir = "input"
output_dir = "output"


def hough_lines_draw(img_out, peaks, rho, theta):
    """Draw lines on an image corresponding to accumulator peaks.
    This method won't be used by the autograder, but you need to implement it to
    get the images required by the problem set.

    Hint:
    Refer to http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    to plot these lines. Notice that the first image (checkerboard) is symmetric in x and y.
    You should try with an asymmetric image to ensure you are plotting the lines properly (i.e. triangle)

    Parameters
    ----------
        img_out: 3-channel (color) image
        peaks: Px2 matrix where each row is a (rho_idx, theta_idx) index pair
        rho: vector of rho values, such that rho[rho_idx] is a valid rho value
        theta: vector of theta values, such that theta[theta_idx] is a valid theta value
    """
    for peak in peaks:
        current_rho = rho[peak[0]]
        current_theta = theta[peak[1]]
        a = np.cos(current_theta)
        b = np.sin(current_theta)
        x0 = a * current_rho
        y0 = b * current_rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_circles(img_in, circles_array):
    """ Draws circles on a given monochrome image.

    Parameters
    ----------
        img_in (numpy.array): monochrome image
        circles_array (numpy.array): numpy array of size n x 3, where n is the number of
                                     non-overlapping circles found by find_circles().
                                     Each row is a (x, y, r) triple parameterizing a circle.

    Returns
    -------

    """
    img_out = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)
    for circle in circles_array:
        cv2.circle(img_out, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 255, 0))

    return img_out


def draw_circles_color(img_in, circles_array):
    """ Draws circles on a given monochrome image.

    Parameters
    ----------
        img_in (numpy.array): monochrome image
        circles_array (numpy.array): numpy array of size n x 3, where n is the number of
                                     non-overlapping circles found by find_circles().
                                     Each row is a (x, y, r) triple parameterizing a circle.

    Returns
    -------

    """
    img_out = img_in
    for circle in circles_array:
        cv2.circle(img_out, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 255, 0))

    return img_out


def main():
    """
    Driver code. Feel free to modify this code as this is to help you get the
    answers for your report. Don't submit this file.
    """

    # 1-a
    # Load the input grayscale image
    img = cv2.imread(os.path.join(input_dir, 'ps2-input0.png'), 0)  # flags=0 ensures grayscale

    filter_size = 13
    threshold1 = 28
    threshold2 = 115


    # # TEST CIRCLE
    # test_circle = cv2.imread(os.path.join(input_dir, 'BasicHoughTest2.png'), 0)
    # test_circle_smoothed = cv2.GaussianBlur(test_circle, (filter_size, filter_size), sigmaX=0, sigmaY=0)  # TODO: call gaussian filter if needed
    # test_circle_edges = cv2.Canny(test_circle_smoothed, threshold1, threshold2)  # TODO: call edge detector function using either the original or the smoothed version
    # H = hough_circles_acc(test_circle_edges, 50, 'single point')
    # cols, rows = H.shape
    # cols = np.linspace(0, cols-1, cols)
    # rows = np.linspace(0, rows-1, rows)
    # H_peaks = hough_peaks(H, Q=5, cols=cols, rows=rows, hough_threshold=100,
    #                       nhood_radii=(5, 5))  # TODO: implement this, try different parameters
    # print H_peaks
    #
    # # There should be only one peak given that there is one circle
    # peak_x = H_peaks[0, 1]  # Select the x value from H_peaks
    # peak_y = H_peaks[0, 0]  # Select the y value from H_peaks
    # circle = np.array([[peak_x, peak_y, 50]])
    #
    # # Draw circles
    # output_image = draw_circles(test_circle, circle)
    # cv2.imwrite(os.path.join(output_dir, 'BasicHoughTest2-ps2-4-a-3.png'), output_image)
    # cv2.imwrite(os.path.join(output_dir, 'BasicHoughTest2-ps2-4-a-1.png'), test_circle_edges)
    #
    # highlighted_peaks = np.copy(H)
    # for peak in H_peaks:
    #     cv2.circle(highlighted_peaks, center=(peak[1], peak[0]), radius=20, color=255)
    # # TODO: Highlight peaks, you could use cv2.circle in each peak
    # cv2.imwrite(os.path.join(output_dir, 'BasicHoughTest2-ps2-4-a-2.png'), highlighted_peaks)
    #
    # # 4-b
    # # TODO: Modifiy hough_circles_acc to work with the 'point plus' method
    # # When using 'point_plus you need to two gradient images, one in x and another in y. See cv2.Sobel
    #
    # test_circle_grad_x = cv2.Sobel(test_circle, cv2.CV_64F, 1, 0, ksize=15)  # X gradient image
    # test_circle_grad_y = cv2.Sobel(test_circle, cv2.CV_64F, 0, 1, ksize=15)  # Y gradient image
    # H = hough_circles_acc(test_circle_edges, 50, 'point plus', grad_x=test_circle_grad_x, grad_y=test_circle_grad_y)
    # cv2.imwrite(os.path.join(output_dir, 'circle_H.png'), H)
    # cols, rows = H.shape
    # cols = np.linspace(0, cols-1, cols)
    # rows = np.linspace(0, rows-1, rows)
    # H_peaks = hough_peaks(H, Q=5, cols=cols, rows=rows, hough_threshold=20, nhood_radii=(5, 5))
    # print H_peaks
    #
    # # There should be only one peak given that there is one circle
    # peak_x = H_peaks[0, 1]  # Select the x value from H_peaks
    # peak_y = H_peaks[0, 0]  # Select the y value from H_peaks
    # circle = np.array([[peak_x, peak_y, 50]])
    #
    # # Draw circles
    # output_image = draw_circles(test_circle, circle)
    # cv2.imwrite(os.path.join(output_dir, 'BasicHoughTest2-ps2-4-b-1.png'), output_image)





    # 8a
    # Image 1
    challenge_image_orig = cv2.imread(os.path.join(input_dir, 'challenge_image1.jpg'), 1)
    challenge_image = cv2.cvtColor(challenge_image_orig, cv2.COLOR_BGR2GRAY)
    challenge_image_smoothed = cv2.GaussianBlur(challenge_image, (7, 7), sigmaX=0, sigmaY=0)
    challenge_image_edges = cv2.Canny(challenge_image_smoothed, 28, 115)
    cv2.imwrite(os.path.join(output_dir, 'challenge_image1_edges2.png'), challenge_image_edges)

    challenge_image_grad_x = cv2.Sobel(challenge_image, cv2.CV_64F, 1, 0, ksize=7)
    challenge_image_grad_y = cv2.Sobel(challenge_image, cv2.CV_64F, 0, 1, ksize=7)
    radii_range = np.linspace(19, 26, 15)
    circles = find_circles(challenge_image_edges, challenge_image_grad_x, challenge_image_grad_y,
                           radii=radii_range, hough_threshold=200, nhood_radii=(20, 20))

    circles = circles.astype(np.int)
    output_image = draw_circles_color(challenge_image_orig, circles)
    for circle in circles:
        text = ''
        if circle[2] in [20, 21]:
            text = 'Penny'
        elif circle[2] in [22, 23]:
            text = 'Nickle'
        elif circle[2] in [19]:
            text = 'Dime'
        elif circle[2] in [25, 26]:
            text = 'Quarter'
        cv2.putText(output_image, text, (circle[0]-circle[2]+2, circle[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-a-1.png'), output_image)


    # Image 2
    challenge_image_orig = cv2.imread(os.path.join(input_dir, 'challenge_image2.jpg'), 1)
    challenge_image = cv2.cvtColor(challenge_image_orig, cv2.COLOR_BGR2GRAY)
    challenge_image_smoothed = cv2.GaussianBlur(challenge_image, (7, 7), sigmaX=0, sigmaY=0)
    challenge_image_edges = cv2.Canny(challenge_image_smoothed, 28, 115)
    cv2.imwrite(os.path.join(output_dir, 'challenge_image2_edges2.png'), challenge_image_edges)

    challenge_image_grad_x = cv2.Sobel(challenge_image, cv2.CV_64F, 1, 0, ksize=7)
    challenge_image_grad_y = cv2.Sobel(challenge_image, cv2.CV_64F, 0, 1, ksize=7)
    radii_range = np.linspace(19, 26, 15)
    circles = find_circles(challenge_image_edges, challenge_image_grad_x, challenge_image_grad_y,
                           radii=radii_range, hough_threshold=200, nhood_radii=(20, 20))

    circles = circles.astype(np.int)
    output_image = draw_circles_color(challenge_image_orig, circles)
    for circle in circles:
        text = ''
        if circle[2] in [20, 21]:
            text = 'Penny'
        elif circle[2] in [22, 23]:
            text = 'Nickle'
        elif circle[2] in [19]:
            text = 'Dime'
        elif circle[2] in [25, 26]:
            text = 'Quarter'
        cv2.putText(output_image, text, (circle[0]-circle[2]+2, circle[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-b-1.png'), output_image)


    # Image 3
    challenge_image_orig = cv2.imread(os.path.join(input_dir, 'challenge_image3.jpg'), 1)
    challenge_image = cv2.cvtColor(challenge_image_orig, cv2.COLOR_BGR2GRAY)
    challenge_image_smoothed = cv2.GaussianBlur(challenge_image, (7, 7), sigmaX=0, sigmaY=0)
    # challenge_image_edges = cv2.Canny(challenge_image_smoothed, 38, 85)
    challenge_image_edges = cv2.Canny(challenge_image_smoothed, 28, 115)
    cv2.imwrite(os.path.join(output_dir, 'challenge_image3_edges2.png'), challenge_image_edges)

    challenge_image_grad_x = cv2.Sobel(challenge_image, cv2.CV_64F, 1, 0, ksize=3)
    challenge_image_grad_y = cv2.Sobel(challenge_image, cv2.CV_64F, 0, 1, ksize=3)
    # radii_range = np.linspace(18, 27, 19)
    radii_range = np.linspace(19, 26, 29)
    circles = find_circles(challenge_image_edges, challenge_image_grad_x, challenge_image_grad_y,
                           radii=radii_range, hough_threshold=200, nhood_radii=(20, 20))

    temp_circles, circles = circles, np.zeros((0, circles.shape[1]))
    for circle in temp_circles:
        color = challenge_image_orig[circle[1], circle[0]]
        if color[0] <= color[1] or color[0] <= color[2]:
            circles = np.concatenate((circles, [circle]))

    circles = circles.astype(np.int)
    output_image = draw_circles_color(challenge_image_orig, circles)
    for circle in circles:
        text = ''
        if circle[2] in [20, 21]:
            text = 'Penny'
        elif circle[2] in [22, 23]:
            text = 'Nickle'
        elif circle[2] in [19]:
            text = 'Dime'
        elif circle[2] in [25, 26]:
            text = 'Quarter'
        cv2.putText(output_image, text, (circle[0]-circle[2]+2, circle[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-c-1.png'), output_image)





    # # TODO: Compute edge image (img_edges). Refer to find_edges.py if you want to use a GUI to test parameters
    # smoothed_img = cv2.GaussianBlur(img, (filter_size, filter_size), sigmaX=0, sigmaY=0)
    # img_edges = cv2.Canny(smoothed_img, threshold1, threshold2)
    # # img_edges = cv2.Canny(img, threshold1, threshold2)
    # cv2.imwrite(os.path.join(output_dir, 'ps2-1-a-1.png'), img_edges)
    #
    # # 2-a
    # # Compute Hough Transform for lines on edge image
    # H, rho, theta = hough_lines_acc(img_edges, rho_res=1, theta_res=math.pi/180)  # TODO: Implement this
    # # TODO: Write a normalized uint8 version, mapping min value to 0 and max to 255
    # cv2.imwrite(os.path.join(output_dir, 'ps2-2-a-1.png'), H)
    #
    # # 2-b
    # # Find peaks (local maxima) in accumulator array
    # peaks = hough_peaks(H, Q=10, cols=rho, rows=theta, hough_threshold=200, nhood_radii=(5, 5))  # TODO: implement this, try different parameters
    #
    # highlighted_peaks = np.copy(H)
    # for peak in peaks:
    #     cv2.circle(highlighted_peaks, center=(peak[1], peak[0]), radius=20, color=255)
    # # TODO: Highlight peaks, you could use cv2.circle in each peak
    # cv2.imwrite(os.path.join(output_dir, 'ps2-2-b-1.png'), highlighted_peaks)
    #
    # # 2-c
    # # Draw lines corresponding to accumulator peaks
    # img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    # hough_lines_draw(img_out, peaks, rho, theta)  # TODO: implement this
    # cv2.imwrite(os.path.join(output_dir, 'ps2-2-c-1.png'), img_out)
    #
    # # 3-a
    # # TODO: Read ps2-input0-noise.png, compute a smoothed image using a Gaussian filter
    # img_noise = cv2.imread(os.path.join(input_dir, 'ps2-input0-noise.png'), 0)
    # img_noise_smoothed = cv2.GaussianBlur(img_noise, (filter_size, filter_size), sigmaX=0, sigmaY=0)  # TODO: call gaussian filter
    # cv2.imwrite(os.path.join(output_dir, 'ps2-3-a-1.png'), img_noise_smoothed)
    #
    # # 3-b
    # # TODO: Compute binary edge images for both original image and smoothed version
    # img_noise_edges = cv2.Canny(img_noise, threshold1, threshold2)  # TODO: call edge detector function
    # img_noise_smoothed_edges = cv2.Canny(img_noise_smoothed, threshold1, threshold2)  # TODO: call edge detector function
    # cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-1.png'), img_noise_edges)
    # cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-2.png'), img_noise_smoothed_edges)
    #
    # # 3-c
    # # TODO: Apply Hough methods to smoothed image, tweak parameters to find best lines
    # H, rho, theta = hough_lines_acc(img_noise_smoothed_edges, rho_res=1, theta_res=math.pi/180)
    # peaks = hough_peaks(H, Q=10, cols=rho, rows=theta, hough_threshold=90, nhood_radii=(5, 5))  # TODO: call hough_peaks
    #
    # highlighted_peaks = np.copy(H)
    # for peak in peaks:
    #     cv2.circle(highlighted_peaks, center=(peak[1], peak[0]), radius=20, color=255)
    # # TODO: Highlight peaks, you could use cv2.circle in each peak
    # cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-1.png'), highlighted_peaks)
    #
    # img_out_noisy = cv2.cvtColor(img_noise, cv2.COLOR_GRAY2BGR)  # TODO: draw lines on the noisy image
    # hough_lines_draw(img_out_noisy, peaks, rho, theta)
    # cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-2.png'), img_out_noisy)
    #
    # # 4-a
    # test_circle = cv2.imread(os.path.join(input_dir, 'test_circle.png'), 0)
    # test_circle_smoothed = cv2.GaussianBlur(test_circle, (filter_size, filter_size), sigmaX=0, sigmaY=0)  # TODO: call gaussian filter if needed
    # test_circle_edges = cv2.Canny(test_circle_smoothed, threshold1, threshold2)  # TODO: call edge detector function using either the original or the smoothed version
    # H = hough_circles_acc(test_circle_edges, 75, 'single point')
    # # cv2.imwrite(os.path.join(output_dir, 'circle_H.png'), H)
    # cols, rows = H.shape
    # cols = np.linspace(0, cols-1, cols)
    # rows = np.linspace(0, rows-1, rows)
    # H_peaks = hough_peaks(H, Q=1, cols=cols, rows=rows, hough_threshold=200,
    #                       nhood_radii=(5, 5))  # TODO: implement this, try different parameters
    #
    # cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-1.png'), test_circle_edges)
    #
    # highlighted_peaks = np.copy(H)
    # for peak in H_peaks:
    #     cv2.circle(highlighted_peaks, center=(peak[1], peak[0]), radius=20, color=255)
    # # TODO: Highlight peaks, you could use cv2.circle in each peak
    # cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-2.png'), highlighted_peaks)
    #
    # # 4-b
    # # TODO: Modifiy hough_circles_acc to work with the 'point plus' method
    # # When using 'point_plus you need to two gradient images, one in x and another in y. See cv2.Sobel
    #
    # test_circle_grad_x = cv2.Sobel(test_circle, cv2.CV_64F, 1, 0, ksize=15)  # X gradient image
    # test_circle_grad_y = cv2.Sobel(test_circle, cv2.CV_64F, 0, 1, ksize=15)  # Y gradient image
    # H = hough_circles_acc(test_circle_edges, 75, 'point plus', grad_x=test_circle_grad_x, grad_y=test_circle_grad_y)
    # cv2.imwrite(os.path.join(output_dir, 'circle_H.png'), H)
    # cols, rows = H.shape
    # cols = np.linspace(0, cols-1, cols)
    # rows = np.linspace(0, rows-1, rows)
    # H_peaks = hough_peaks(H, Q=1, cols=cols, rows=rows, hough_threshold=40, nhood_radii=(5, 5))
    #
    # # There should be only one peak given that there is one circle
    # peak_x = H_peaks[0, 1]  # Select the x value from H_peaks
    # peak_y = H_peaks[0, 0]  # Select the y value from H_peaks
    # circle = np.array([[peak_x, peak_y, 75]])
    #
    # # Draw circles
    # output_image = draw_circles(test_circle, circle)
    # cv2.imwrite(os.path.join(output_dir, 'ps2-4-b-1.png'), output_image)
    #
    # # From this point on you will only use the 'point plus' method
    # # 5-a
    # pens_and_coins = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)
    # pens_and_coins_smoothed = cv2.GaussianBlur(pens_and_coins, (filter_size, filter_size), sigmaX=0, sigmaY=0)
    # pens_and_coins_edges = cv2.Canny(pens_and_coins_smoothed, threshold1, threshold2)
    # cv2.imwrite(os.path.join(output_dir, 'pens_and_coins_edges_old.png'), pens_and_coins_edges)
    #
    # H, rho, theta = hough_lines_acc(pens_and_coins_edges, rho_res=1, theta_res=math.pi/180)
    # peaks = hough_peaks(H, Q=10, cols=rho, rows=theta, hough_threshold=100, nhood_radii=(5, 5))
    # print(peaks)
    #
    # highlighted_peaks = np.copy(H)
    # for peak in peaks:
    #     cv2.circle(highlighted_peaks, center=(peak[1], peak[0]), radius=20, color=255)
    # # TODO: Highlight peaks, you could use cv2.circle in each peak
    # cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-1.png'), highlighted_peaks)
    #
    # img_out = cv2.cvtColor(pens_and_coins, cv2.COLOR_GRAY2BGR)
    # hough_lines_draw(img_out, peaks, rho, theta)  # TODO: implement this
    # cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-2.png'), img_out)
    #
    # # 5-b
    # pens_and_coins_smoothed = cv2.GaussianBlur(pens_and_coins, (filter_size, filter_size), sigmaX=0, sigmaY=0)  # In case you need different parameters
    # pens_and_coins_edges = cv2.Canny(pens_and_coins_smoothed, threshold1, threshold2)  # In case you need different parameters
    #
    # pens_and_coins_grad_x = cv2.Sobel(pens_and_coins, cv2.CV_64F, 1, 0, ksize=11)
    # pens_and_coins_grad_y = cv2.Sobel(pens_and_coins, cv2.CV_64F, 0, 1, ksize=11)
    # cv2.imwrite(os.path.join(output_dir, 'pens_and_coins_edges.png'), pens_and_coins_edges)
    # cv2.imwrite(os.path.join(output_dir, 'pens_and_coins_grad_x.png'), pens_and_coins_grad_x)
    # cv2.imwrite(os.path.join(output_dir, 'pens_and_coins_grad_y.png'), pens_and_coins_grad_y)
    # H = hough_circles_acc(pens_and_coins_edges, 20, 'point plus',
    #                       grad_x=pens_and_coins_grad_x, grad_y=pens_and_coins_grad_y)
    # cv2.imwrite(os.path.join(output_dir, 'pens_and_coins_h.png'), H)
    # cols, rows = H.shape
    # cols = np.linspace(0, cols-1, cols)
    # rows = np.linspace(0, rows-1, rows)
    # H_peaks = hough_peaks(H, Q=20, cols=cols, rows=rows, hough_threshold=100, nhood_radii=(20, 20))
    # print(H_peaks)
    #
    # # There could be more than one peak returned. Now draw all the non-overlapping circles
    # peaks_x = H_peaks[:, 1]  # Peaks x values
    # peaks_y = H_peaks[:, 0]   # Peaks y values
    # peaks_r = [20] * len(peaks_x)
    # circles = np.column_stack((peaks_x, peaks_y, peaks_r))
    #
    # # Draw circles
    # output_image = draw_circles(pens_and_coins, circles)
    # cv2.imwrite(os.path.join(output_dir, 'ps2-5-b-1.png'), output_image)
    #
    # # 5-c
    # # find_circles should call hough_circles_acc and hough_peaks.
    # pens_and_coins_smoothed = cv2.GaussianBlur(pens_and_coins, (filter_size, filter_size), sigmaX=0, sigmaY=0)  # In case you need different parameters
    # pens_and_coins_edges = cv2.Canny(pens_and_coins_smoothed, threshold1, threshold2)  # In case you need different parameters
    #
    # pens_and_coins_grad_x = cv2.Sobel(pens_and_coins, cv2.CV_64F, 1, 0, ksize=3)
    # pens_and_coins_grad_y = cv2.Sobel(pens_and_coins, cv2.CV_64F, 0, 1, ksize=3)
    # cv2.imwrite(os.path.join(output_dir, 'pens_and_coins_edges.png'), pens_and_coins_edges)
    # cv2.imwrite(os.path.join(output_dir, 'pens_and_coins_grad_x.png'), pens_and_coins_grad_x)
    # cv2.imwrite(os.path.join(output_dir, 'pens_and_coins_grad_y.png'), pens_and_coins_grad_y)
    # # radii_range = range(10, 50, 1)  # Try different values
    # radii_range = np.linspace(11, 30, 20, endpoint=True)
    # circles = find_circles(pens_and_coins_edges, pens_and_coins_grad_x, pens_and_coins_grad_y,
    #                        radii=radii_range, hough_threshold=220, nhood_radii=(10, 10))
    # print('circles', circles)
    #
    # output_image = draw_circles(pens_and_coins, circles)
    # cv2.imwrite(os.path.join(output_dir, 'ps2-5-c-1.png'), output_image)
    #
    # # 6
    # cluttered_image = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'), 0)
    # # TODO: Identify the lines and circles in the cluttered image
    # # TODO: save image ps2-6-a-1.png
    # cluttered_image_smoothed = cv2.GaussianBlur(cluttered_image, (filter_size, filter_size), sigmaX=0, sigmaY=0)
    # cluttered_image_edges = cv2.Canny(cluttered_image_smoothed, threshold1, threshold2)
    # cv2.imwrite(os.path.join(output_dir, 'cluttered_image_edges.png'), cluttered_image_edges)
    #
    # # 6-a
    # H, rho, theta = hough_lines_acc(cluttered_image_edges, rho_res=1, theta_res=math.pi/180)
    # peaks = hough_peaks(H, Q=14, cols=rho, rows=theta, hough_threshold=100, nhood_radii=(20, 20))
    # print(peaks)
    #
    # highlighted_peaks = np.copy(H)
    # for peak in peaks:
    #     cv2.circle(highlighted_peaks, center=(peak[1], peak[0]), radius=20, color=255)
    # # TODO: Highlight peaks, you could use cv2.circle in each peak
    # cv2.imwrite(os.path.join(output_dir, 'cluttered_image_peaks.png'), highlighted_peaks)
    #
    # img_out = cv2.cvtColor(cluttered_image_smoothed, cv2.COLOR_GRAY2BGR)
    # hough_lines_draw(img_out, peaks, rho, theta)  # TODO: implement this
    # cv2.imwrite(os.path.join(output_dir, 'ps2-6-a-1.png'), img_out)
    #
    #
    # # TODO: save image ps2-6-b-1.png
    # cluttered_image_smoothed = cv2.GaussianBlur(cluttered_image, (filter_size, filter_size), sigmaX=0, sigmaY=0)
    # cluttered_image_edges = cv2.Canny(cluttered_image_smoothed, threshold1, threshold2)
    # cv2.imwrite(os.path.join(output_dir, 'cluttered_image_edges.png'), cluttered_image_edges)
    #
    # # 6-b
    # H, rho, theta = hough_lines_acc(cluttered_image_edges, rho_res=1, theta_res=math.pi/180)
    # peaks = hough_peaks(H, Q=14, cols=rho, rows=theta, hough_threshold=100, nhood_radii=(20, 20))
    # print(peaks)
    #
    # highlighted_peaks = np.copy(H)
    # for peak in peaks:
    #     cv2.circle(highlighted_peaks, center=(peak[1], peak[0]), radius=20, color=255)
    # # TODO: Highlight peaks, you could use cv2.circle in each peak
    # cv2.imwrite(os.path.join(output_dir, 'cluttered_image_peaks.png'), highlighted_peaks)
    #
    # img_out = cv2.cvtColor(cluttered_image_smoothed, cv2.COLOR_GRAY2BGR)
    # hough_lines_draw(img_out, peaks, rho, theta)  # TODO: implement this
    # cv2.imwrite(os.path.join(output_dir, 'ps2-6-b-1.png'), img_out)
    #
    #
    # # TODO: save image ps2-6-c-1.png
    # cluttered_image_smoothed = cv2.GaussianBlur(cluttered_image, (3, 3), sigmaX=0, sigmaY=0)
    # cluttered_image_edges = cv2.Canny(cluttered_image_smoothed, 28, 115)
    # cv2.imwrite(os.path.join(output_dir, 'cluttered_image_edges.png'), cluttered_image_edges)
    #
    # # 6-c
    # cluttered_image_grad_x = cv2.Sobel(cluttered_image, cv2.CV_64F, 1, 0, ksize=3)
    # cluttered_image_grad_y = cv2.Sobel(cluttered_image, cv2.CV_64F, 0, 1, ksize=3)
    # radii_range = np.linspace(25, 35, 21)
    # circles = find_circles(cluttered_image_edges, cluttered_image_grad_x, cluttered_image_grad_y,
    #                        radii=radii_range, hough_threshold=156, nhood_radii=(20, 20))
    # print('circles', circles)
    #
    # output_image = draw_circles(cluttered_image, circles)
    # cv2.imwrite(os.path.join(output_dir, 'ps2-6-c-1.png'), output_image)



    # TODO: Don't forget to answer questions 7 and 8 in your report
    pass


if __name__ == '__main__':
    main()
