"""Problem Set 2: Edges and Lines"""
import os
from ps2 import *

# I/O directories
input_dir = "input"
output_dir = "output"


def draw_circles(img_in, circles_array):
    """Draws circles on a given monochrome image.

    No changes are needed in this function.

    Note that OpenCV's cv2.circle( ) function requires the center point to be defined using the (x, y)
    coordinate convention.

    Args:
        img_in (numpy.array): monochrome image
        circles_array (numpy.array): numpy array of size n x 3, where n is the number of
                                     circles found by find_circles(). Each row is a (x, y, r)
                                     triple that parametrizes a circle.

    Returns:
        numpy.array: 3-channel image with circles drawn.
    """

    img_out = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)
    return draw_circles_color(img_out, circles_array)


def draw_circles_color(img_in, circles_array):
    img_out = img_in
    for circle in circles_array:
        cv2.circle(img_out, (int(circle[1]), int(circle[0])), circle[2], (0, 255, 0))

    return img_out


def hough_lines_draw(img_in, peaks, rho, theta):
    """Draws lines on an image corresponding to accumulator peaks.

    This method won't be used by the autograder, but you need to implement it to
    get the images required by the problem set.

    Hint:
    Refer to http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    to plot these lines. Notice that the first image (checkerboard) is symmetric in x and y.
    You should try with an asymmetric image to ensure you are plotting the lines properly (i.e. triangle)

    Note that OpenCV's cv2.line( ) function requires points to be defined using the (x, y) coordinate convention.

    If the input image is a 2D array you should convert it to BGR, see cv2.cvtColor( )

    Args:
        img_in (numpy.array): input image
        peaks (numpy.array): array containing the local maxima in the Hough accumulator where each row is
                             a pair of [row_id, col_id] pair
        rho (numpy.array): vector of rho values, one for each row of H.
        theta (numpy.array): vector of theta values in the range [0,pi), one for each column of H.

    Returns:
        numpy.array: 3-channel image with lines drawn.
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

        cv2.line(img_in, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_in


def highlight_peaks(H, peaks):
    """Returns a version of H with the best peaks highlighted.

    This function may use cv2.circle to
    plot in color circles centered in each best peak. Alternative methods are accepted.

    If the input image is a 2D array you should convert it to BGR, see cv2.cvtColor( )

    Args:
        H (numpy.array): H accumulator array (usually normalized and scaled)
        peaks (numpy.array): array containing the local maxima in the Hough accumulator where each row is
                             a pair of [row_id, col_id] pair

    Returns:
        numpy.array: 3-channel version of H.
    """
    highlighted_peaks = H.copy()
    for peak in peaks:
        cv2.circle(highlighted_peaks, center=(peak[1], peak[0]), radius=20, color=255)
    return highlighted_peaks


def get_edge_image(img_in, threshold1=28, threshold2=115):
    """Calls an edge function of your choice, i.e. cv2.Canny.

    You may modify this function's signature if you
    want to include more parameters. This includes removing **kwargs and adding named parameters.

    Args:
        img_in (numpy.array): input image.
        threshold1: threshold 1
        threshold2: threshold 2

    Returns:
        numpy.array: edge image.
    """
    return cv2.Canny(img_in, threshold1, threshold2)


def get_smoothed_image(img_in, ksize=(13, 13), sigmaX=0, sigmaY=0):
    """Returns a smoothed version of img_in after using a Gaussian filter You may modify this function's signature
    if you want to include more parameters. This includes removing **kwargs and adding named parameters.

    Args:
        img_in (numpy.array): input image.
        ksize: kernel size
        sigmaX: sigma x
        sigmaY: sigma y

    Returns:
        numpy.array: edge image.
    """
    return cv2.GaussianBlur(img_in, ksize, sigmaX=sigmaX, sigmaY=sigmaY)


def normalize_and_scale(img_in):
    """Maps values in img_in to fit in the range [0, 255]. This will be usually called before displaying or
    saving an image. You may use cv2.normalize or create your own.

    Args:
        img_in (numpy.array): input image.

    Returns:
        numpy.array: output image with pixel values in [0, 255]
    """
    return cv2.normalize(img_in, 0, 255, norm_type=cv2.NORM_MINMAX)


def part_1(save_imgs=True):

    # 1-a
    # Load the input grayscale image
    img = cv2.imread(os.path.join(input_dir, 'ps2-input0.png'), 0)  # flags=0 ensures grayscale

    img_edges = get_edge_image(img)

    if save_imgs:
        cv2.imwrite(os.path.join(output_dir, 'ps2-1-a-1.png'), img_edges)

    return {"img": img, "img_edges": img_edges}


def part_2():

    # 2-a
    # Compute Hough Transform for lines on edge image
    p1 = part_1(False)
    img_edges = p1["img_edges"]
    rho_res = 1  # You may have to try different values
    theta_res = np.pi/180  # You may have to try different values
    H, rho, theta = hough_lines_acc(img_edges, rho_res, theta_res)

    # Write a normalized uint8 version of H, mapping min value from 0 to 255
    H_n = normalize_and_scale(H)

    cv2.imwrite(os.path.join(output_dir, 'ps2-2-a-1.png'), H_n)

    # 2-b
    # Find peaks (local maxima) in accumulator array
    hough_threshold = 200  # You may have to try different values
    nhood_delta = (5, 5)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta, cols=rho, rows=theta)

    highlighted_peaks = highlight_peaks(H_n, peaks)

    cv2.imwrite(os.path.join(output_dir, 'ps2-2-b-1.png'), highlighted_peaks)

    # 2-c
    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(p1["img"], cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    img_out = hough_lines_draw(img_out, peaks, rho, theta)

    cv2.imwrite(os.path.join(output_dir, 'ps2-2-c-1.png'), img_out)


def part_3():

    # 3-a
    # Read ps2-input0-noise.png, compute a smoothed image using a Gaussian filter
    img_noise = cv2.imread(os.path.join(input_dir, 'ps2-input0-noise.png'), 0)
    img_noise_smoothed = get_smoothed_image(img_noise)

    cv2.imwrite(os.path.join(output_dir, 'ps2-3-a-1.png'), img_noise_smoothed)

    # 3-b
    # Compute binary edge images for both original image and smoothed version
    img_noise_edges = get_edge_image(img_noise)
    img_noise_smoothed_edges = get_edge_image(img_noise_smoothed)

    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-1.png'), img_noise_edges)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-2.png'), img_noise_smoothed_edges)

    # 3-c
    # Apply Hough methods to the smoothed image, tweak parameters to find best lines
    rho_res = 1  # You may have to try different values
    theta_res = np.pi / 180  # You may have to try different values
    H, rho, theta = hough_lines_acc(img_noise_smoothed_edges, rho_res, theta_res)

    hough_threshold = 180  # You may have to try different values
    nhood_delta = (5, 5)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta, cols=rho, rows=theta)

    H_n = normalize_and_scale(H)
    highlighted_peaks = highlight_peaks(H_n, peaks)

    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-1.png'), highlighted_peaks)

    img_noise = cv2.cvtColor(img_noise, cv2.COLOR_GRAY2BGR)
    img_out_noisy = hough_lines_draw(img_noise, peaks, rho, theta)

    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-2.png'), img_out_noisy)


def part_4a():

    # 4-a
    test_circle = cv2.imread(os.path.join(input_dir, 'test_circle.png'), 0)
    test_circle_smoothed = get_smoothed_image(test_circle)  # If needed
    test_circle_edges = get_edge_image(test_circle_smoothed)  # You can use the smoothed image instead

    radius = 75
    H = hough_circles_acc(test_circle, test_circle_edges, radius, False)

    hough_threshold = 200  # You may have to try different values
    nhood_delta = (1, 1)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta)

    cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-1.png'), test_circle_edges)

    H_n = normalize_and_scale(H)
    highlighted_peaks = highlight_peaks(H_n, peaks)

    cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-2.png'), highlighted_peaks)


def part_4b():

    # 4-b
    test_circle = cv2.imread(os.path.join(input_dir, 'test_circle.png'), 0)
    test_circle_smoothed = get_smoothed_image(test_circle)  # If needed
    test_circle_edges = get_edge_image(test_circle_smoothed)  # You can use the smoothed image instead

    # Use hough_circles_acc with the 'point plus' method
    radius = 75
    H = hough_circles_acc(test_circle, test_circle_edges, radius)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-b-1-H.png'), H)

    hough_threshold = 100  # You may have to try different values
    nhood_delta = (1, 1)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta)

    # There should be only one peak given that there is one circle
    circle = np.array([[peaks[0, 0], peaks[0, 1], radius]])

    # Draw circles
    output_image = draw_circles(test_circle, circle)

    cv2.imwrite(os.path.join(output_dir, 'ps2-4-b-1.png'), output_image)


def part_5a():

    # From this point on you will only use the 'point plus' method
    # 5-a
    pens_and_coins = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)
    pens_and_coins_smoothed = get_smoothed_image(pens_and_coins)  # If needed
    pens_and_coins_edges = get_edge_image(pens_and_coins_smoothed)  # You can use the smoothed image instead

    rho_res = 1  # You may have to try different values
    theta_res = np.pi / 180  # You may have to try different values
    H, rho, theta = hough_lines_acc(pens_and_coins_edges, rho_res, theta_res)

    hough_threshold = 190  # You may have to try different values
    nhood_delta = (1, 1)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta)

    H_n = normalize_and_scale(H)
    highlighted_peaks = highlight_peaks(H_n, peaks)

    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-1.png'), highlighted_peaks)

    img_out = cv2.cvtColor(pens_and_coins, cv2.COLOR_GRAY2BGR)
    img_out = hough_lines_draw(img_out, peaks, rho, theta)

    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-2.png'), img_out)


def part_5b():

    # 5-b
    pens_and_coins = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)
    pens_and_coins_smoothed = get_smoothed_image(pens_and_coins)  # If needed
    pens_and_coins_edges = get_edge_image(pens_and_coins_smoothed)  # You can use the smoothed image instead

    radius = 23  # Fixed radius DO NOT change it. It will be used for grading.
    H = hough_circles_acc(pens_and_coins, pens_and_coins_edges, radius)

    hough_threshold = 55  # You may have to try different values
    nhood_delta = (20, 20)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta)

    # There could be more than one peak returned. Now draw all the non-overlapping circles
    circles = np.column_stack((peaks[:, 0], peaks[:, 1], [radius] * len(peaks)))

    # Draw circles
    output_image = draw_circles(pens_and_coins, circles)

    cv2.imwrite(os.path.join(output_dir, 'ps2-5-b-1.png'), output_image)


def part_5c():
    # 5-c
    pens_and_coins = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)
    pens_and_coins_smoothed = get_smoothed_image(pens_and_coins)  # If needed
    pens_and_coins_edges = get_edge_image(pens_and_coins_smoothed)  # You can use the smoothed image instead

    # radii = range(10, 50, 1)  # Try different values
    radii = np.linspace(16, 30, 15, endpoint=True)
    hough_threshold = 150  # You may have to try different values
    nhood_delta = (20, 20)  # You may have to try different values
    circles = find_circles(pens_and_coins, pens_and_coins_edges, radii, hough_threshold, nhood_delta)

    output_image = draw_circles(pens_and_coins, circles)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-c-1.png'), output_image)


def part_6():
    """Finds and plots lines and circles in the image ps2-input2.png. This part should accurately detect
    the pens and coins. For more information follow the problem set documentation.

    The images to be saved are:
    - ps2-6-a-1.png
    - ps2-6-b-1.png
    - ps2-6-c-1.png

    Returns:
        None.
    """

    cluttered_image = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'), 0)

    cluttered_image_smoothed = get_smoothed_image(cluttered_image)
    cluttered_image_edges = get_edge_image(cluttered_image_smoothed)

    # 6a
    rho_res = 1
    theta_res = np.pi / 180
    H, rho, theta = hough_lines_acc(cluttered_image_edges, rho_res, theta_res)

    hough_threshold = 110
    nhood_delta = (10, 10)
    peaks = hough_peaks(H, hough_threshold, nhood_delta)

    img_out = cv2.cvtColor(cluttered_image, cv2.COLOR_GRAY2BGR)
    img_out = hough_lines_draw(img_out, peaks, rho, theta)

    cv2.imwrite(os.path.join(output_dir, 'ps2-6-a-1.png'), img_out)

    # 6c
    cluttered_image_smoothed = get_smoothed_image(cluttered_image, ksize=(3, 3))
    cluttered_image_edges = get_edge_image(cluttered_image_smoothed, threshold1=28, threshold2=115)
    cv2.imwrite(os.path.join(output_dir, 'ps2-6-c-1-cluttered_image_edges.png'), cluttered_image_edges)

    # radii = np.linspace(23, 35, 25, endpoint=True)
    radii = np.linspace(24, 35, 45)
    hough_threshold = 186  # You may have to try different values
    nhood_delta = (10, 10)  # You may have to try different values
    circles = find_circles(cluttered_image, cluttered_image_edges, radii, hough_threshold, nhood_delta)

    output_image = draw_circles(cluttered_image, circles)
    cv2.imwrite(os.path.join(output_dir, 'ps2-6-c-1.png'), output_image)


def part_8_1():
    # Image 1
    challenge_image_orig = cv2.imread(os.path.join(input_dir, 'challenge_image1.jpg'), 1)
    challenge_image = cv2.cvtColor(challenge_image_orig, cv2.COLOR_BGR2GRAY)
    challenge_image_smoothed = get_smoothed_image(challenge_image, (3, 3), sigmaX=0, sigmaY=0)
    challenge_image_edges = get_edge_image(challenge_image_smoothed, 28, 115)
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-a-1-challenge_image1_edges2.png'), challenge_image_edges)

    radii_range = np.linspace(19, 26, 15)
    circles = find_circles(challenge_image, challenge_image_edges, radii=radii_range, hough_threshold=150, nhood_delta=(10, 10))

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
        cv2.putText(output_image, text, (circle[1]-circle[2]+2, circle[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-a-1.png'), output_image)


def part_8_2():
    # Image 2
    challenge_image_orig = cv2.imread(os.path.join(input_dir, 'challenge_image2.jpg'), 1)
    challenge_image = cv2.cvtColor(challenge_image_orig, cv2.COLOR_BGR2GRAY)
    challenge_image_smoothed = get_smoothed_image(challenge_image, (3, 3), sigmaX=0, sigmaY=0)
    challenge_image_edges = get_edge_image(challenge_image_smoothed, 28, 115)
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-b-1-challenge_image2_edges2.png'), challenge_image_edges)

    radii_range = np.linspace(19, 26, 15)
    circles = find_circles(challenge_image, challenge_image_edges, radii=radii_range, hough_threshold=150, nhood_delta=(10, 10))

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
        cv2.putText(output_image, text, (circle[1]-circle[2]+2, circle[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-b-1.png'), output_image)


def part_8_3():
    # Image 3
    challenge_image_orig = cv2.imread(os.path.join(input_dir, 'challenge_image3.jpg'), 1)
    challenge_image = cv2.cvtColor(challenge_image_orig, cv2.COLOR_BGR2GRAY)
    challenge_image_smoothed = get_smoothed_image(challenge_image, (7, 7), sigmaX=0, sigmaY=0)
    # challenge_image_edges = get_edge_image(challenge_image_smoothed, 38, 85)
    challenge_image_edges = get_edge_image(challenge_image_smoothed, 28, 115)
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-c-1-challenge_image3_edges2.png'), challenge_image_edges)

    # radii_range = np.linspace(18, 27, 19)
    radii_range = np.linspace(19, 26, 29)
    circles = find_circles(challenge_image, challenge_image_edges, radii=radii_range, hough_threshold=150, nhood_delta=(10, 10))

    temp_circles, circles = circles, np.zeros((0, circles.shape[1]))
    for circle in temp_circles:
        color = challenge_image_orig[int(circle[0]), int(circle[1])]
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
        cv2.putText(output_image, text, (circle[1]-circle[2]+2, circle[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, 'ps2-8-c-1.png'), output_image)


if __name__ == '__main__':
    # part_1()
    # part_2()
    # part_3()
    # part_4a()
    # part_4b()
    # part_5a()
    # part_5b()
    # part_5c()
    part_6()
    # TODO: Don't forget to answer part 7 in your report
    # part_8_1()
    # part_8_2()
    # part_8_3()
