"""Problem Set 2: Edges and Lines"""
import os
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
    pass  # TODO: Your code here (nothing to return, just draw on img_out directly)


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


def main():
    """
    Driver code. Feel free to modify this code as this is to help you get the
    answers for your report. Don't submit this file.
    """

    # 1-a
    # Load the input grayscale image
    img = cv2.imread(os.path.join(input_dir, 'ps2-input0.png'), 0)  # flags=0 ensures grayscale

    # TODO: Compute edge image (img_edges). Refer to find_edges.py if you want to use a GUI to test parameters
    img_edges = None  # TODO: Add the edge function of your choice, i.e. cv2.Canny
    cv2.imwrite(os.path.join(output_dir, 'ps2-1-a-1.png'), img_edges)

    # 2-a
    # Compute Hough Transform for lines on edge image
    H, rho, theta = hough_lines_acc(img_edges, rho_res=None, theta_res=None)  # TODO: Implement this
    # TODO: Write a normalized uint8 version, mapping min value to 0 and max to 255
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-a-1.png'), H)

    # 2-b
    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, Q=None, cols=rho, rows=theta, hough_threshold=None, nhood_radii=(5, 5))  # TODO: implement this, try different parameters

    highlighted_peaks = None
    # TODO: Highlight peaks, you could use cv2.circle in each peak
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-b-1.png'), highlighted_peaks)

    # 2-c
    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, peaks, rho, theta)  # TODO: implement this
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-c-1.png'), img_out)

    # 3-a
    # TODO: Read ps2-input0-noise.png, compute a smoothed image using a Gaussian filter
    img_noise = cv2.imread(os.path.join(input_dir, 'ps2-input0-noise.png'), 0)
    img_noise_smoothed = None  # TODO: call gaussian filter
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-a-1.png'), img_noise_smoothed)

    # 3-b
    # TODO: Compute binary edge images for both original image and smoothed version
    img_noise_edges = None  # TODO: call edge detector function
    img_noise_smoothed_edges = None  # TODO: call edge detector function
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-1.png'), img_noise_edges)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-2.png'), img_noise_smoothed_edges)

    # 3-c
    # TODO: Apply Hough methods to smoothed image, tweak parameters to find best lines
    H, rho, theta = hough_lines_acc(img_edges, rho_res=None, theta_res=None)
    peaks = None  # TODO: call hough_peaks
    highlighted_peaks = None
    # TODO: Highlight peaks, you could use cv2.circle in each peak
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-1.png'), highlighted_peaks)

    img_out_noisy = None  # TODO: draw lines on the noisy image
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-2.png'), img_out_noisy)

    # 4-a
    test_circle = cv2.imread(os.path.join(input_dir, 'test_circle.png'), 0)
    test_circle_smoothed = None  # TODO: call gaussian filter if needed
    test_circle_edges = None  # TODO: call edge detector function using either the original or the smoothed version
    H = hough_circles_acc(test_circle_edges, 75, 'single point')
    cols, rows = H.shape
    H_peaks = hough_peaks(H, Q=None, cols=cols, rows=rows, hough_threshold=None,
                          nhood_radii=(5, 5))  # TODO: implement this, try different parameters
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-1.png'), test_circle_edges)

    highlighted_peaks = None
    # TODO: Highlight peaks, you could use cv2.circle in each peak
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-2.png'), highlighted_peaks)

    # 4-b
    # TODO: Modifiy hough_circles_acc to work with the 'point plus' method
    # When using 'point_plus you need to two gradient images, one in x and another in y. See cv2.Sobel

    test_circle_grad_x = None  # X gradient image
    test_circle_grad_y = None  # Y gradient image
    H = hough_circles_acc(test_circle_edges, 75, 'point plus', grad_x=test_circle_grad_x, grad_y=test_circle_grad_y)
    cols, rows = H.shape
    H_peaks = hough_peaks(H, Q=None, cols=cols, rows=rows, hough_threshold=None, nhood_radii=(None, None))

    # There should be only one peak given that there is one circle
    peak_x = None  # Select the x value from H_peaks
    peak_y = None  # Select the y value from H_peaks
    circle = np.array([[peak_x, peak_y, 75]])

    # Draw circles
    output_image = draw_circles(test_circle, circle)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-b-1.png'), output_image)

    # From this point on you will only use the 'point plus' method
    # 5-a
    pens_and_coins = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)
    pens_and_coins_smoothed = None
    pens_and_coins_edges = None

    H, rho, theta = hough_lines_acc(pens_and_coins_edges, rho_res=None, theta_res=None)
    peaks = hough_peaks(H, Q=None, cols=rho, rows=theta, hough_threshold=None, nhood_radii=(None, None))

    highlighted_peaks = None
    # TODO: Highlight peaks, you could use cv2.circle in each peak
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-1.png'), highlighted_peaks)

    img_out = cv2.cvtColor(pens_and_coins, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)  # TODO: implement this
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-2.png'), img_out)

    # 5-b
    pens_and_coins_smoothed = None  # In case you need different parameters
    pens_and_coins_edges = None  # In case you need different parameters

    pens_and_coins_grad_x = None
    pens_and_coins_grad_y = None
    H = hough_circles_acc(pens_and_coins_edges, 20, 'point plus',
                          grad_x=pens_and_coins_grad_x, grad_y=pens_and_coins_grad_y)
    cols, rows = H.shape
    H_peaks = hough_peaks(H, Q=None, cols=cols, rows=rows, hough_threshold=None, nhood_radii=(None, None))

    # There could be more than one peak returned. Now draw all the non-overlapping circles
    peaks_x = [None]  # Peaks x values
    peaks_y = [None]  # Peaks y values
    peaks_r = [20] * len(peaks_x)
    circles = np.column_stack((peaks_x, peaks_y, peaks_r))

    # Draw circles
    output_image = draw_circles(pens_and_coins, circles)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-b-1.png'), output_image)

    # 5-c
    # find_circles should call hough_circles_acc and hough_peaks.
    pens_and_coins_smoothed = None  # In case you need different parameters
    pens_and_coins_edges = None  # In case you need different parameters

    pens_and_coins_grad_x = None  # X gradient image
    pens_and_coins_grad_y = None  # Y gradient image
    radii_range = range(20, 50, 5)  # Try different values
    circles = find_circles(pens_and_coins_edges, pens_and_coins_grad_x, pens_and_coins_grad_y,
                           radii=radii_range, hough_threshold=None, nhood_radii=None)

    output_image = draw_circles(pens_and_coins, circles)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-c-1.png'), output_image)

    # 6
    cluttered_image = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'), 0)
    # TODO: Identify the lines and circles in the cluttered image
    # TODO: save image ps2-6-a-1.png
    # TODO: save image ps2-6-b-1.png
    # TODO: save image ps2-6-c-1.png

    # TODO: Don't forget to answer questions 7 and 8 in your report
    pass


if __name__ == '__main__':
    main()
