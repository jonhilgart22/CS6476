from ps3 import *
import cv2

class SigmaParamFinder():
    def __init__(self, imageL, imageR):
        self.imgL = imageL
        self.imgR = imageR
        self.sigmaL = 0
        self.sigmaR = 0

        def onchangeSigmaL(sigma):
            self.sigmaL = max(sigma,0) / 100.0
            self._render()
        def onchangeSigmaR(sigma):
            self.sigmaR = max(sigma,0) / 100.0
            self._render()

        cv2.namedWindow('disparity')
        cv2.createTrackbar('sigmaL', 'disparity', self.sigmaL, 1000, onchangeSigmaL)
        cv2.createTrackbar('sigmaR', 'disparity', self.sigmaR, 1000, onchangeSigmaR)

        self._render()

        print "Adjust the parameters as desired.  Hit any key to close."

        cv2.waitKey(0)

        cv2.destroyWindow('disparity')
        cv2.destroyWindow('sigmaL')
        cv2.destroyWindow('sigmaR')
    def sigmaLGet(self):
        return self.sigmaL
    def sigmaRGet(self):
        return self.sigmaR
    def _render(self):
        if self.sigmaL > 0:
            self._smoothed_imgL = add_noise(self.imgL,self.sigmaL)
            cv2.imshow('sigmaL', self._smoothed_imgL)
        else:
            cv2.imshow('sigmaL', self.imgL)

        if self.sigmaR > 0:
            self._smoothed_imgR = add_noise(self.imgR, self.sigmaR)
            cv2.imshow('sigmaR', self._smoothed_imgR)
        else:
            cv2.imshow('sigmaR', self.imgR)

