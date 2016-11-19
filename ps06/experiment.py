"""Problem Set 6: Optic Flow"""
import cv2
import os

from ps6 import *

# I/O directories
input_dir = "input"
output_dir = "output"


# Utility code
def quiver(U, V, stride, scale, color=(0, 255, 0)):
    img_out = np.zeros((V.shape[0], U.shape[1], 3), dtype=np.uint8)
    for y in xrange(0, V.shape[0], stride):
        for x in xrange(0, U.shape[1], stride):
            cv2.line(img_out, (x, y), (x + int(U[y, x]*scale), y+int(V[y, x]*scale)), color, 1)
            cv2.circle(img_out,  (x + int(U[y, x]*scale), y+int(V[y, x]*scale)), 1, color, 1)
    return img_out


def jet_colormap(disp_img):
    return cv2.applyColorMap(normalize_and_scale(disp_img).astype(np.uint8), cv2.COLORMAP_JET)


# Driver code
def main():
    """
    Driver code. Feel free to modify this code as this is to help you get the
    answers for your report.
    """

    # 1a
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.
    ShiftR2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR2.png'), 0) / 255.
    ShiftR5U5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR5U5.png'), 0) / 255.
    ShiftR10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.
    ShiftR20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0) / 255.
    ShiftR40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0) / 255.

    kernel = np.ones((25, 25)) / (25 ** 2)
    Shift0_blur = cv2.filter2D(Shift0, -1, kernel)
    ShiftR2_blur = cv2.filter2D(ShiftR2, -1, kernel)
    ShiftR5U5_blur = cv2.filter2D(ShiftR5U5, -1, kernel)
    ShiftR10_blur = cv2.filter2D(ShiftR10, -1, kernel)
    ShiftR20_blur = cv2.filter2D(ShiftR20, -1, kernel)
    ShiftR40_blur = cv2.filter2D(ShiftR40, -1, kernel)

    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    k_type = 'uniform'  # TODO: Choose between "uniform" or "gaussian"
    k_size = 25  # TODO: Define a size for a square kernel
    U, V = optic_flow_LK(Shift0_blur, ShiftR2_blur, k_size, k_type)  # TODO: implement this

    # TODO: Save U, V as side-by-side false-color image or single quiver plot:
    jet_U = jet_colormap(U)
    jet_V = jet_colormap(V)
    U_V = np.concatenate((jet_U, jet_V), axis=1)  # TODO: place jet_U and jet_V side by side if you choose to save a false-color image
    cv2.imwrite(os.path.join(output_dir, "ps6-1-a-1.png"), U_V)  # Jet colormap

    # Use the following two lines if you instead choose to display your results as
    # a flow image.
    disp_img = quiver(U, V, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-1-a-1.quiver.png"), disp_img)  # Quiver plot

    # TODO: Similarly for Shift0 and ShiftR5U5. You may try smoothing the input images
    # Shift0_blur = cv2.GaussianBlur(Shift0, (5, 5), 11.)
    # ShiftR5U5_blur = cv2.GaussianBlur(ShiftR5U5, (5, 5), 11.)
    # kernel = np.ones((25, 25)) / (25 ** 2)
    # Shift0_blur = cv2.filter2D(Shift0, -1, kernel)
    # ShiftR5U5_blur = cv2.filter2D(ShiftR5U5, -1, kernel)
    k_type = 'uniform'  # TODO: Choose between "uniform" or "gaussian"
    k_size = 45  # TODO: Define a size for a square kernel
    U, V = optic_flow_LK(Shift0_blur, ShiftR5U5_blur, k_size, k_type)

    # TODO: save ps6-1-a-2.png
    jet_U = jet_colormap(U)
    jet_V = jet_colormap(V)
    U_V = np.concatenate((jet_U, jet_V), axis=1)  # TODO: place jet_U and jet_V side by side if you choose to save a false-color image
    cv2.imwrite(os.path.join(output_dir, "ps6-1-a-2.png"), U_V)  # Jet colormap

    disp_img = quiver(U, V, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-1-a-2.quiver.png"), disp_img)  # Quiver plot

    # 1b
    # TODO: Similarly for ShiftR10, ShiftR20 and ShiftR40.
    # You can use different k_type and k_size for each.
    # TODO: save ps6-1-b-1.png
    # Shift0_blur = cv2.GaussianBlur(Shift0, (5, 5), 11.)
    # ShiftR10_blur = cv2.GaussianBlur(ShiftR10, (5, 5), 1.)
    # kernel = np.ones((25, 25)) / (25 ** 2)
    # Shift0_blur = cv2.filter2D(Shift0, -1, kernel)
    # ShiftR10_blur = cv2.filter2D(ShiftR10, -1, kernel)
    k_type = 'uniform'  # TODO: Choose between "uniform" or "gaussian"
    k_size = 55  # TODO: Define a size for a square kernel
    U, V = optic_flow_LK(Shift0_blur, ShiftR10_blur, k_size, k_type)

    jet_U = jet_colormap(U)
    jet_V = jet_colormap(V)
    U_V = np.concatenate((jet_U, jet_V), axis=1)  # TODO: place jet_U and jet_V side by side if you choose to save a false-color image
    cv2.imwrite(os.path.join(output_dir, "ps6-1-b-1.png"), U_V)  # Jet colormap
    disp_img = quiver(U, V, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-1-b-1.quiver.png"), disp_img)  # Quiver plot

    # TODO: save ps6-1-b-2.png
    # Shift0_blur = cv2.GaussianBlur(Shift0, (55, 55), 11.)
    # ShiftR20_blur = cv2.GaussianBlur(ShiftR20, (55, 55), 11.)
    # kernel = np.ones((25, 25)) / (25 ** 2)
    # Shift0_blur = cv2.filter2D(Shift0, -1, kernel)
    # ShiftR20_blur = cv2.filter2D(ShiftR20, -1, kernel)
    k_type = 'uniform'  # TODO: Choose between "uniform" or "gaussian"
    k_size = 75  # TODO: Define a size for a square kernel
    U, V = optic_flow_LK(Shift0_blur, ShiftR20_blur, k_size, k_type)

    jet_U = jet_colormap(U)
    jet_V = jet_colormap(V)
    U_V = np.concatenate((jet_U, jet_V), axis=1)  # TODO: place jet_U and jet_V side by side if you choose to save a false-color image
    cv2.imwrite(os.path.join(output_dir, "ps6-1-b-2.png"), U_V)  # Jet colormap
    disp_img = quiver(U, V, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-1-b-2.quiver.png"), disp_img)  # Quiver plot

    # TODO: save ps6-1-b-3.png
    k_type = 'uniform'  # TODO: Choose between "uniform" or "gaussian"
    k_size = 75  # TODO: Define a size for a square kernel
    U, V = optic_flow_LK(Shift0_blur, ShiftR40_blur, k_size, k_type)

    jet_U = jet_colormap(U)
    jet_V = jet_colormap(V)
    U_V = np.concatenate((jet_U, jet_V), axis=1)  # TODO: place jet_U and jet_V side by side if you choose to save a false-color image
    cv2.imwrite(os.path.join(output_dir, "ps6-1-b-3.png"), U_V)  # Jet colormap
    disp_img = quiver(U, V, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-1-b-3.quiver.png"), disp_img)  # Quiver plot

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, levels)  # TODO: implement this
    yos_img_01_g_pyr_img = create_combined_img(yos_img_01_g_pyr)  # TODO: implement this
    cv2.imwrite(os.path.join(output_dir, "ps6-2-a-1.png"), yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = laplacian_pyramid(yos_img_01_g_pyr)  # TODO: implement this
    yos_img_01_l_pyr_img = create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps6-2-b-1.png"), yos_img_01_l_pyr_img)

    k_size = 15
    k_type = 'uniform'

    # 3a
    levels = 4  # Define the number of levels
    yos_img_02_g_pyr = gaussian_pyramid(yos_img_02, levels)
    # TODO: Select appropriate pyramid *level* that leads to best optic flow estimation
    level_id = 3
    U, V = optic_flow_LK(yos_img_01_g_pyr[level_id], yos_img_02_g_pyr[level_id],
                         k_size, k_type)  # You may use different k_size and k_type
    # TODO: Scale up U, V to original image size (note: don't forget to scale values as well!)
    for level in xrange(level_id):
        U = expand(U) * 2
        V = expand(V) * 2

    yos_img_02_h, yos_img_02_w = yos_img_02.shape
    U = U[:yos_img_02_h, :yos_img_02_w]
    V = V[:yos_img_02_h, :yos_img_02_w]
    # y1, x1 = (np.array(U.shape) - np.array(yos_img_02.shape)) / 2.
    # y2, x2 = np.array(U.shape) - [y1, x1]
    # U = U[y1:y2, x1:x2]
    # V = V[y1:y2, x1:x2]

    scale = 10  # define a scale value
    stride = 3  # define a stride value
    yos_img_01_02_flow = quiver(U, V, scale, stride)  # This image will be saved later
    yos_img_02_warped = warp(yos_img_02, U, V)  # TODO: implement this

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    # Note: Scale values such that zero difference maps to neutral gray, max -ve to black and max +ve to white

    # Similarly, compute displacements for yos_img_02 and yos_img_03
    levels = 4  # Define a number of levels
    yos_img_03_g_pyr = gaussian_pyramid(yos_img_03, levels)
    level_id = 3
    U, V = optic_flow_LK(yos_img_02_g_pyr[level_id], yos_img_03_g_pyr[level_id],
                         k_size, k_type)  # You may use different k_size and k_type
    # TODO: Scale up U, V to original image size (note: don't forget to scale values as well!)
    for level in xrange(level_id):
        U = expand(U) * 2
        V = expand(V) * 2

    yos_img_03_h, yos_img_03_w = yos_img_03.shape
    U = U[:yos_img_03_h, :yos_img_03_w]
    V = V[:yos_img_03_h, :yos_img_03_w]
    # y1, x1 = (np.array(U.shape) - np.array(yos_img_03.shape)) / 2.
    # y2, x2 = np.array(U.shape) - [y1, x1]
    # U = U[y1:y2, x1:x2]
    # V = V[y1:y2, x1:x2]

    scale = 10  # define a scale value
    stride = 3  # define a stride value
    yos_img_02_03_flow = quiver(U, V, scale, stride)

    yos_img_03_warped = warp(yos_img_03, U, V)
    diff_yos_img_02_03 = yos_img_02 - yos_img_03_warped

    cv2.imwrite(os.path.join(output_dir, "ps6-3-a-1.png"),
                np.concatenate((yos_img_01_02_flow, yos_img_02_03_flow), axis=0))
    cv2.imwrite(os.path.join(output_dir, "ps6-3-a-2.png"),
                np.concatenate((normalize_and_scale(diff_yos_img_01_02),
                                normalize_and_scale(diff_yos_img_02_03)), axis=0))

    # TODO: Repeat for DataSeq2 (save images)
    dtsq2_01 = cv2.imread(os.path.join(input_dir, 'DataSeq2', '0.png'), 0) / 255.
    dtsq2_02 = cv2.imread(os.path.join(input_dir, 'DataSeq2', '1.png'), 0) / 255.
    dtsq2_03 = cv2.imread(os.path.join(input_dir, 'DataSeq2', '2.png'), 0) / 255.

    # TODO: save ps6-3-a-3.png
    # TODO: save ps6-3-a-4.png
    levels = 4
    dtsq2_01_g_pyr = gaussian_pyramid(dtsq2_01, levels)  # TODO: implement this
    dtsq2_01_g_pyr_img = create_combined_img(dtsq2_01_g_pyr)  # TODO: implement this
    # cv2.imwrite(os.path.join(output_dir, "ps6-3-b-1.png"), dtsq2_01_g_pyr_img)

    dtsq2_01_l_pyr = laplacian_pyramid(dtsq2_01_g_pyr)  # TODO: implement this
    dtsq2_01_l_pyr_img = create_combined_img(dtsq2_01_l_pyr)
    # cv2.imwrite(os.path.join(output_dir, "ps6-3-b-2.png"), dtsq2_01_l_pyr_img)

    k_size = 15
    k_type = 'uniform'

    levels = 4  # Define the number of levels
    dtsq2_02_g_pyr = gaussian_pyramid(dtsq2_02, levels)
    # TODO: Select appropriate pyramid *level* that leads to best optic flow estimation
    level_id = 2
    U, V = optic_flow_LK(dtsq2_01_g_pyr[level_id], dtsq2_02_g_pyr[level_id],
                         k_size, k_type)  # You may use different k_size and k_type
    # TODO: Scale up U, V to original image size (note: don't forget to scale values as well!)
    for level in xrange(level_id):
        U = expand(U) * 2
        V = expand(V) * 2

    dtsq2_02_h, dtsq2_02_w = dtsq2_02.shape
    U = U[:dtsq2_02_h, :dtsq2_02_w]
    V = V[:dtsq2_02_h, :dtsq2_02_w]
    # y1, x1 = (np.array(U.shape) - np.array(dtsq2_02.shape)) / 2.
    # y2, x2 = np.array(U.shape) - [y1, x1]
    # U = U[y1:y2, x1:x2]
    # V = V[y1:y2, x1:x2]

    scale = 10  # define a scale value
    stride = 3  # define a stride value
    dtsq2_01_02_flow = quiver(U, V, scale, stride)  # This image will be saved later
    dtsq2_02_warped = warp(dtsq2_02, U, V)  # TODO: implement this

    diff_dtsq2_01_02 = dtsq2_01 - dtsq2_02_warped
    # Note: Scale values such that zero difference maps to neutral gray, max -ve to black and max +ve to white

    # Similarly, compute displacements for dtsq2_02 and dtsq2_03
    levels = 4  # Define a number of levels
    dtsq2_03_g_pyr = gaussian_pyramid(dtsq2_03, levels)
    level_id = 2
    U, V = optic_flow_LK(dtsq2_02_g_pyr[level_id], dtsq2_03_g_pyr[level_id],
                         k_size, k_type)  # You may use different k_size and k_type
    # TODO: Scale up U, V to original image size (note: don't forget to scale values as well!)
    for level in xrange(level_id):
        U = expand(U) * 2
        V = expand(V) * 2

    dtsq2_03_h, dtsq2_03_w = dtsq2_03.shape
    U = U[:dtsq2_03_h, :dtsq2_03_w]
    V = V[:dtsq2_03_h, :dtsq2_03_w]
    # y1, x1 = (np.array(U.shape) - np.array(dtsq2_03.shape)) / 2.
    # y2, x2 = np.array(U.shape) - [y1, x1]
    # U = U[y1:y2, x1:x2]
    # V = V[y1:y2, x1:x2]

    scale = 10  # define a scale value
    stride = 3  # define a stride value
    dtsq2_02_03_flow = quiver(U, V, scale, stride)

    dtsq2_03_warped = warp(dtsq2_03, U, V)
    diff_dtsq2_02_03 = dtsq2_02 - dtsq2_03_warped

    cv2.imwrite(os.path.join(output_dir, "ps6-3-a-3.png"),
                np.concatenate((dtsq2_01_02_flow, dtsq2_02_03_flow), axis=0))
    cv2.imwrite(os.path.join(output_dir, "ps6-3-a-4.png"),
                np.concatenate((normalize_and_scale(diff_dtsq2_01_02),
                                normalize_and_scale(diff_dtsq2_02_03)), axis=0))

    # 4a
    ShiftR10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.
    ShiftR20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0) / 255.
    ShiftR40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0) / 255.

    levels = 4  # Define the number of levels
    k_size = 25  # Set a kernel size
    k_type = 'uniform'  # Set a kernel type
    sigma = 1.  # You may use a different value
    U10, V10 = hierarchical_LK(Shift0, ShiftR10, levels, k_size, k_type, sigma)  # TODO: implement this
    jet_U10 = jet_colormap(U10)
    jet_V10 = jet_colormap(V10)

    disp_img = quiver(U10, V10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-a-1.10.png"), disp_img)  # Quiver plot

    U20, V20 = hierarchical_LK(Shift0, ShiftR20, levels, k_size, k_type, sigma)  # You may try different values
    jet_U20 = jet_colormap(U20)
    jet_V20 = jet_colormap(V20)

    disp_img = quiver(U20, V20, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-a-1.20.png"), disp_img)  # Quiver plot

    U40, V40 = hierarchical_LK(Shift0, ShiftR40, levels, k_size, k_type, sigma)  # You may try different values
    jet_U40 = jet_colormap(U40)
    jet_V40 = jet_colormap(V40)

    disp_img = quiver(U40, V40, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-a-1.40.png"), disp_img)  # Quiver plot

    U10_V10 = np.concatenate((jet_U10, jet_V10), axis=1)
    U20_V20 = np.concatenate((jet_U20, jet_V20), axis=1)
    U40_V40 = np.concatenate((jet_U40, jet_V40), axis=1)

    # TODO: Save displacement image pairs (U, V), stacked
    UV_stacked = np.concatenate((U10_V10, U20_V20, U40_V40), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-a-1.png"), UV_stacked)

    # Save difference between each warped image and original image (Shift0), stacked
    ShiftR10_warped = warp(ShiftR10, U10, V10)
    ShiftR20_warped = warp(ShiftR20, U20, V20)
    ShiftR40_warped = warp(ShiftR40, U40, V40)

    diff_0_10 = ShiftR10_warped - Shift0
    diff_0_20 = ShiftR20_warped - Shift0
    diff_0_40 = ShiftR40_warped - Shift0

    # TODO: Generate and save ps6-4-a-2.png. Hint: You can use np.concatenate()
    # TODO: save ps6-4-a-2.png
    warped_stacked = np.concatenate((normalize_and_scale(diff_0_10),
                                     normalize_and_scale(diff_0_20),
                                     normalize_and_scale(diff_0_40)), axis=0)
    cv2.imwrite(os.path.join(output_dir, 'ps6-4-a-2.png'), warped_stacked)

    # 4b
    # TODO: Repeat for DataSeq1 (use yos_img_01.png as the base image)
    # TODO: save ps6-4-b-1.png
    levels = 4  # Define the number of levels
    k_size = 151  # Set a kernel size
    k_type = 'uniform'  # Set a kernel type
    sigma = 1.  # You may use a different value
    U1, V1 = hierarchical_LK(yos_img_01, yos_img_02, levels, k_size, k_type, sigma)  # TODO: implement this
    jet_U1 = jet_colormap(U1)
    jet_V1 = jet_colormap(V1)

    disp_img = quiver(U1, V1, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-b-1.1.png"), disp_img)  # Quiver plot

    U2, V2 = hierarchical_LK(yos_img_02, yos_img_03, levels, k_size, k_type, sigma)  # You may try different values
    jet_U2 = jet_colormap(U2)
    jet_V2 = jet_colormap(V2)

    disp_img = quiver(U2, V2, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-b-1.2.png"), disp_img)  # Quiver plot

    U1_V1 = np.concatenate((jet_U1, jet_V1), axis=1)
    U2_V2 = np.concatenate((jet_U2, jet_V2), axis=1)

    UV_stacked = np.concatenate((U1_V1, U2_V2), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-b-1.png"), UV_stacked)

    # TODO: save ps6-4-b-2.png
    yos_img_02_warped = warp(yos_img_02, U1, V1)
    yos_img_03_warped = warp(yos_img_03, U2, V2)

    diff_1_2 = yos_img_02_warped - yos_img_01
    diff_2_3 = yos_img_03_warped - yos_img_01

    warped_stacked = np.concatenate((normalize_and_scale(diff_1_2),
                                     normalize_and_scale(diff_2_3)), axis=0)
    cv2.imwrite(os.path.join(output_dir, 'ps6-4-b-2.png'), warped_stacked)

    # 4c
    # TODO: Repeat for DataSeq1 (use 0.png as the base image)
    # TODO: save ps6-4-c-1.png
    levels = 7  # Define the number of levels
    k_size = 27  # Set a kernel size
    k_type = 'uniform'  # Set a kernel type
    sigma = 1.  # You may use a different value
    U1, V1 = hierarchical_LK(dtsq2_01, dtsq2_02, levels, k_size, k_type, sigma)  # TODO: implement this
    jet_U1 = jet_colormap(U1)
    jet_V1 = jet_colormap(V1)

    disp_img = quiver(U1, V1, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-c-1.1.png"), disp_img)  # Quiver plot

    U2, V2 = hierarchical_LK(dtsq2_02, dtsq2_03, levels, k_size, k_type, sigma)  # You may try different values
    jet_U2 = jet_colormap(U2)
    jet_V2 = jet_colormap(V2)

    disp_img = quiver(U2, V2, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-c-1.2.png"), disp_img)  # Quiver plot

    U1_V1 = np.concatenate((jet_U1, jet_V1), axis=1)
    U2_V2 = np.concatenate((jet_U2, jet_V2), axis=1)

    UV_stacked = np.concatenate((U1_V1, U2_V2), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-4-c-1.png"), UV_stacked)

    # TODO: save ps6-4-c-2.png
    dtsq2_02_warped = warp(dtsq2_02, U1, V1)
    dtsq2_03_warped = warp(dtsq2_03, U2, V2)

    diff_1_2 = dtsq2_02_warped - dtsq2_01
    diff_2_3 = dtsq2_03_warped - dtsq2_01

    warped_stacked = np.concatenate((normalize_and_scale(diff_1_2),
                                     normalize_and_scale(diff_2_3)), axis=0)
    cv2.imwrite(os.path.join(output_dir, 'ps6-4-c-2.png'), warped_stacked)

    # 5a
    # This part is more difficult. Try to be creative on how you can achieve
    # good results using optic flow.
    juggle_01 = cv2.imread(os.path.join(input_dir, 'Juggle', '0.png'), 0) / 255.
    juggle_02 = cv2.imread(os.path.join(input_dir, 'Juggle', '1.png'), 0) / 255.
    juggle_03 = cv2.imread(os.path.join(input_dir, 'Juggle', '2.png'), 0) / 255.

    kernel = gaussianKernel(75, 9.)
    juggle_01 = cv2.filter2D(juggle_01, -1, kernel)
    juggle_02 = cv2.filter2D(juggle_02, -1, kernel)
    juggle_03 = cv2.filter2D(juggle_03, -1, kernel)

    # TODO: save ps6-5-a-1.png
    levels = 3  # Define the number of levels
    k_size = 75  # Set a kernel size
    k_type = 'gaussian'  # Set a kernel type
    sigma = 9.  # You may use a different value
    U1, V1 = hierarchical_LK(juggle_01, juggle_02, levels, k_size, k_type, sigma)  # TODO: implement this
    jet_U1 = jet_colormap(U1)
    jet_V1 = jet_colormap(V1)

    disp_img = quiver(U1, V1, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-5-a-1.1.png"), disp_img)  # Quiver plot

    U2, V2 = hierarchical_LK(juggle_02, juggle_03, levels, k_size, k_type, sigma)  # You may try different values
    jet_U2 = jet_colormap(U2)
    jet_V2 = jet_colormap(V2)

    disp_img = quiver(U2, V2, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps6-5-a-1.2.png"), disp_img)  # Quiver plot

    U1_V1 = np.concatenate((jet_U1, jet_V1), axis=1)
    U2_V2 = np.concatenate((jet_U2, jet_V2), axis=1)

    UV_stacked = np.concatenate((U1_V1, U2_V2), axis=0)
    cv2.imwrite(os.path.join(output_dir, "ps6-5-a-1.png"), UV_stacked)

    # TODO: save ps6-5-a-2.png
    juggle_02_warped = warp(juggle_02, U1, V1)
    juggle_03_warped = warp(juggle_03, U2, V2)

    diff_1_2 = juggle_02_warped - juggle_01
    diff_2_3 = juggle_03_warped - juggle_01

    jet_diff_1_2 = jet_colormap(diff_1_2)
    jet_diff_2_3 = jet_colormap(diff_2_3)
    warped_stacked = np.concatenate((jet_diff_1_2, jet_diff_2_3), axis=0)
    cv2.imwrite(os.path.join(output_dir, 'ps6-5-a-2.png'), warped_stacked)

if __name__ == "__main__":
    main()
