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

    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    k_type = None  # TODO: Choose between "uniform" or "gaussian"
    k_size = None  # TODO: Define a size for a square kernel
    U, V = optic_flow_LK(Shift0, ShiftR2, k_size, k_type)  # TODO: implement this

    # TODO: Save U, V as side-by-side false-color image or single quiver plot:
    # jet_U = jet_colormap(U)
    # jet_V = jet_colormap(V)
    # U_V = None  # TODO: place jet_U and jet_V side by side if you choose to save a false-color image
    # cv2.imwrite(os.path.join(output_dir, "ps6-1-a-1.png"), U_V)  # Jet colormap

    # Use the following two lines if you instead choose to display your results as
    # a flow image.
    # disp_img = quiver(U, V, scale=3, stride=10)
    # cv2.imwrite(os.path.join(output_dir, "ps6-1-a-1.png"), disp_img)  # Quiver plot
    # TODO: save ps6-1-a-1.png

    # TODO: Similarly for Shift0 and ShiftR5U5. You may try smoothing the input images
    k_type = None  # TODO: Choose between "uniform" or "gaussian"
    k_size = None  # TODO: Define a size for a square kernel
    U, V = optic_flow_LK(Shift0, ShiftR5U5, k_size, k_type)
    # TODO: save ps6-1-a-2.png

    # 1b
    # TODO: Similarly for ShiftR10, ShiftR20 and ShiftR40.
    # You can use different k_type and k_size for each.
    # TODO: save ps6-1-b-1.png
    # TODO: save ps6-1-b-2.png
    # TODO: save ps6-1-b-3.png

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

    # 3a
    levels = None  # Define the number of levels
    yos_img_02_g_pyr = gaussian_pyramid(yos_img_02, levels)
    # TODO: Select appropriate pyramid *level* that leads to best optic flow estimation
    level_id = None
    U, V = optic_flow_LK(yos_img_01_g_pyr[level_id], yos_img_02_g_pyr[level_id],
                         k_size, k_type)  # You may use different k_size and k_type
    # TODO: Scale up U, V to original image size (note: don't forget to scale values as well!)

    scale = None  # define a scale value
    stride = None  # define a stride value
    yos_img_01_02_flow = quiver(U, V, scale, stride)  # This image will be saved later
    yos_img_02_warped = warp(yos_img_02, U, V)  # TODO: implement this

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    # Note: Scale values such that zero difference maps to neutral gray, max -ve to black and max +ve to white

    # Similarly, compute displacements for yos_img_02 and yos_img_03
    levels = None  # Define a number of levels
    yos_img_03_g_pyr = gaussian_pyramid(yos_img_03, levels)
    level_id = None
    U, V = optic_flow_LK(yos_img_02_g_pyr[level_id], yos_img_03_g_pyr[level_id],
                         k_size, k_type)  # You may use different k_size and k_type
    # TODO: Scale up U, V to original image size (note: don't forget to scale values as well!)
    scale = None  # define a scale value
    stride = None  # define a stride value
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

    # 4a
    ShiftR10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.
    ShiftR20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0) / 255.
    ShiftR40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0) / 255.

    levels = None  # Define the number of levels
    k_size = None  # Set a kernel size
    k_type = None  # Set a kernel type
    sigma = 1.  # You may use a different value
    U10, V10 = hierarchical_LK(Shift0, ShiftR10, levels, k_size, k_type, sigma)  # TODO: implement this
    jet_U10 = jet_colormap(U10)
    jet_V10 = jet_colormap(V10)

    U20, V20 = hierarchical_LK(Shift0, ShiftR20, levels, k_size, k_type, sigma)  # You may try different values
    jet_U20 = jet_colormap(U20)
    jet_V20 = jet_colormap(V20)

    U40, V40 = hierarchical_LK(Shift0, ShiftR40, levels, k_size, k_type, sigma)  # You may try different values
    jet_U40 = jet_colormap(U40)
    jet_V40 = jet_colormap(V40)

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

    # 4b
    # TODO: Repeat for DataSeq1 (use yos_img_01.png as the base image)
    # TODO: save ps6-4-b-1.png
    # TODO: save ps6-4-b-2.png

    # 4c
    # TODO: Repeat for DataSeq1 (use 0.png as the base image)
    # TODO: save ps6-4-c-1.png
    # TODO: save ps6-4-c-2.png

    # 5a
    # This part is more difficult. Try to be creative on how you can achieve
    # good results using optic flow.
    juggle_04 = cv2.imread(os.path.join(input_dir, 'Juggle', '0.png'))
    juggle_05 = cv2.imread(os.path.join(input_dir, 'Juggle', '1.png'))
    juggle_06 = cv2.imread(os.path.join(input_dir, 'Juggle', '2.png'))

    # TODO: save ps6-5-a-1.png
    # TODO: save ps6-5-a-2.png

if __name__ == "__main__":
    main()