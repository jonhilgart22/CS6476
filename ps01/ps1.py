import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extractRed(image):
    """ Returns the red channel of the input image.
    """
    pass


def extractGreen(image):
    """ Returns the green channel of the input image.
    """
    pass


def extractBlue(image):
    """ Returns the blue channel of the input image.
    """
    pass   


def swapGreenBlue(image):
    """ Returns an image with the green and blue channels of the input image swapped.
    """
    pass


def copyPasteMiddle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst.

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of 
        the src into the range [1:2,1:2] of the dst.
    """
    pass


def imageStats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    """
    pass


def normalized(image, stddev):
    """Returns an image with the same mean as the original but with values
    scaled about the mean so as to have a standard deviation of stddev.

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to 
    a float64 type before passing in an image.
    """
    pass


def shiftImageLeft(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.
        
        The returned image has the same shape as the original with 
        the BORDER_REPLICATE rule to fill-in missing values.  See
        
        http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

        for further explanation.
    """
    pass


def differenceImage(img1, img2):
    """Returns the normalized value of the difference between the two input images.
    """
    pass


def addNoise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to 
    channel (0-2).  The parameter sigma controls the standard deviation of the noise.

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to 
    a float64 type before passing in an image.
    """
    pass
