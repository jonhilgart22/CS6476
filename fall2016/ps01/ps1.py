import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extractRed(image):
    """ Returns the red channel of the input image.
    """
    # We can also use split e.g. cv2.split(image)[2]
    return image[:, :, 2]


def extractGreen(image):
    """ Returns the green channel of the input image.
    """
    # We can also use split e.g. cv2.split(image)[1]
    return image[:, :, 1]


def extractBlue(image):
    """ Returns the blue channel of the input image.
    """
    # We can also use split e.g. cv2.split(image)[0]
    return image[:, :, 0]


def swapGreenBlue(image):
    """ Returns an image with the green and blue channels of the input image swapped.
    """
    output = np.zeros(image.shape, dtype=image.dtype)
    output[:, :, :] = image  # deep copy
    output[:, :, 0] = extractGreen(image)
    output[:, :, 1] = extractBlue(image)
    return output


def copyPasteMiddle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst.

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of 
        the src into the range [1:2,1:2] of the dst.
    """
    if len(src.shape) != 2 or len(dst.shape) != 2 or len(shape) != 2:
        return dst

    sx, sy = math.floor((src.shape[0] - shape[0]) / 2), math.floor((src.shape[1] - shape[1]) / 2)
    dx, dy = math.floor((dst.shape[0] - shape[0]) / 2), math.floor((dst.shape[1] - shape[1]) / 2)
    sx_end, sy_end = sx + shape[0], sy + shape[1]
    dx_end, dy_end = dx + shape[0], dy + shape[1]

    output = np.zeros(dst.shape, dtype=dst.dtype)
    output[:, :] = dst  # deep copy
    # copy center of src with size shape to dst
    output[dx:dx_end, dy:dy_end] = src[sx:sx_end, sy:sy_end]

    return output


def imageStats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    """
    image_min = np.min(image)
    image_max = np.max(image)
    image_mean = np.mean(image)
    image_stddev = np.std(image)
    return image_min, image_max, image_mean, image_stddev


def normalized(image, stddev):
    """Returns an image with the same mean as the original but with values
    scaled about the mean so as to have a standard deviation of stddev.

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to 
    a float64 type before passing in an image.
    """
    image_min, image_max, image_mean, image_stddev = imageStats(image)
    output = (((image - image_mean) / image_stddev) * stddev) + image_mean
    return output.astype(np.uint8)


def shiftImageLeft(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.
        
        The returned image has the same shape as the original with 
        the BORDER_REPLICATE rule to fill-in missing values.  See
        
        http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

        for further explanation.
    """
    # move image to the left by "shift" amount and replicate to fill the missing column
    return cv2.copyMakeBorder(image[:, shift:], 0, 0, 0, shift, cv2.BORDER_REPLICATE)


def differenceImage(img1, img2):
    """Returns the normalized value of the difference between the two input images.
    """
    output = img1.astype(np.float) - img2.astype(np.float)
    image_min, image_max, image_mean, image_stddev = imageStats(output)
    # normalize the difference to 0-255
    output = (output + np.abs(image_min)) * ((2 ** 8 - 1) / (np.abs(image_max) + np.abs(image_min)))
    return output.astype(np.uint8)


def addNoise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to 
    channel (0-2).  The parameter sigma controls the standard deviation of the noise.

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to 
    a float64 type before passing in an image.
    """
    noise = np.random.randn(*image.shape) * sigma
    output = np.zeros(image.shape, dtype=image.dtype)
    output[:, :, :] = image
    output[:, :, channel] += noise[:, :, channel]
    return output
