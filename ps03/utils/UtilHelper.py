import cv2
import os
import numpy as np

from guiutils import ParamsFinder
from guiUtilSigma import SigmaParamFinder

def main():
    input_dir = "input"
    L = cv2.imread(os.path.join(input_dir, 'pair1-L.png'),0) / 255.  # grayscale, scale to [0.0, 1.0]
    R = cv2.imread(os.path.join(input_dir, 'pair1-R.png'),0) / 255.

    height, width = L.shape
    minimizeBy = 4 # what fraction of the image should we use
    smallL = cv2.resize(L, (height / minimizeBy, width / minimizeBy), interpolation=cv2.INTER_AREA)
    smallR = cv2.resize(R, (height / minimizeBy, width / minimizeBy), interpolation=cv2.INTER_AREA)

    cv2.imshow('inputL', smallL)
    cv2.imshow('inputR', smallR)

    paramFinder = ParamsFinder(smallL,smallR,L,R,minimizeBy)

    print "Parameters:"
    print "GaussianBlur Filter Size: %f" % paramFinder.filterSize()
    print "w_size %f" % paramFinder.wsize()
    print "dmax %f" % paramFinder.dmaxParam()
    print "SSD ", paramFinder.FuncParam()
    print "sigmaL %f" % paramFinder.getSigmaL()
    print "sigmaR %f" % paramFinder.getSigmaR()

    cv2.destroyAllWindows()

def mainSigmaMover():
    input_dir = "input"
    L = cv2.imread(os.path.join(input_dir, 'pair1-L.png'), 0) / 255.  # grayscale, scale to [0.0, 1.0]
    R = cv2.imread(os.path.join(input_dir, 'pair1-R.png'), 0) / 255.
    cv2.imshow('inputL', L)
    cv2.imshow('inputR', R)

    paramFinder = SigmaParamFinder(L, R)

    print "Parameters:"
    print "sigmaL %f" % paramFinder.sigmaLGet()
    print "sigmaR %f" % paramFinder.sigmaRGet()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() # pick one of these to run
    # mainSigmaMover()

