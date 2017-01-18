import cv2
import numpy as np
from ps3 import *
class ParamsFinder:
    def __init__(self, imageL,imageR,imageLBig,imageRBig,minBy):
        self.imageL = imageL
        self.imageR = imageR
        self.imageLBig = imageLBig
        self.imageRBig = imageRBig
        self._filter_size = 1
        self.dmax = 18
        self.w_size = 8
        self.SSD = False # SSD or NCorr
        self.sigmaL = 0
        self.sigmaR = 0
        self.minBy = minBy
        self.contrast = 0

        def onchangeFilterSize(pos):
            self._filter_size = pos
            self._filter_size += (self._filter_size + 1) % 2
            self._render()
        def onchangedmax(dmax):
            self.dmax = max(dmax,1)
            self._render()
        def onchangew_size(w_size):
            self.w_size = max(w_size,1)
            self._render()
        def onchangeSSD(SSD):
            self.SSD = bool(SSD % 2)
            self._render()
        def onchangeSigmaL(sigma):
            self.sigmaL = sigma
            self._render()
        def onchangeSigmaR(sigma):
            self.sigmaR = sigma
            self._render()
        def onchangeContrast(constrast):
            self.constrast = constrast
            self._render()
        def onchangeShowBig(_):
            self._renderLarge()

        cv2.namedWindow('disparity')
        cv2.createTrackbar('filter_size', 'disparity', self._filter_size, 50, onchangeFilterSize)
        cv2.createTrackbar('dmax','disparity',self.dmax,250,onchangedmax)
        cv2.createTrackbar('w_size', 'disparity', self.w_size, 50, onchangew_size)
        cv2.createTrackbar('Ncorr(0) or SSD(1)', 'disparity', int(self.SSD), 1, onchangeSSD)
        cv2.createTrackbar('sigmaL','disparity',self.sigmaL,100,onchangeSigmaL)
        cv2.createTrackbar('sigmaR', 'disparity', self.sigmaR, 1000, onchangeSigmaR)
        cv2.createTrackbar('showBig (just click on this)','disparity', 0,0,onchangeShowBig)
        cv2.createTrackbar('contrast','disparity',10,20,onchangeContrast)
        self._render()

        print "Adjust the parameters as desired.  Hit any key to close."

        cv2.waitKey(0)

        cv2.destroyWindow('disparity')
        cv2.destroyWindow('smoothedL')
        cv2.destroyWindow('smoothedR')
        cv2.destroyWindow('L')
        cv2.destroyWindow('R')
    def threshold1(self):
        return self._threshold1

    def threshold2(self):
        return self._threshold2

    def filterSize(self):
        return self._filter_size

    def wsize(self):
        return self.w_size
    def dmaxParam(self):
        return self.dmax
    def FuncParam(self):
        return self.SSD
    def getSigmaL(self):
        return self.sigmaL
    def getSigmaR(self):
        return self.sigmaR

    def _renderLarge(self):
	# this only shows left image currently, its a bit slow to do both
        smoothed = cv2.GaussianBlur(self.imageLBig, (self._filter_size,self._filter_size),sigmaX=0,sigmaY=0)
        if self.sigmaL > 0:
            imageL = add_noise(smoothed,self.sigmaL / 10.0)
        if self.contrast > 0:
	    imageL = increase_contrast(smoothed,self.contrast)

        cv2.imshow('bigL',imageL)
        if self.SSD:
            f = disparity_ssd
        else:
            f = disparity_ncorr

        D_L = f(imageL,self.imageRBig,1,self.w_size*self.minBy,self.dmax*self.minBy*2)
        # D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # cv2.imshow('disparityMap',disparity)
        cv2.imshow('bigLDisp', D_L)
    def _render(self):
        # this only shows left image currently, its a bit slow to do both
        self._smoothed_imgL = cv2.GaussianBlur(self.imageL, (self._filter_size, self._filter_size), sigmaX=0, sigmaY=0)
        self._smoothed_imgR = cv2.GaussianBlur(self.imageR, (self._filter_size, self._filter_size), sigmaX=0, sigmaY=0)
        if self.sigmaL > 0:
            self._smoothed_imgL = add_noise(self._smoothed_imgL,self.sigmaL / 10.0)
        if self.contrast > 0:
	    self._smoothed_imgL = increase_contrast(self._smoothed_imgL,self.contrast)
        if self.sigmaR > 0:
            self._smoothed_imgR = add_noise(self._smoothed_imgR,self.sigmaR / 1000.0)
        cv2.imshow('smoothedL', self._smoothed_imgL)
        cv2.imshow('smoothedR', self._smoothed_imgR)
        if self.SSD:
            f = disparity_ssd
        else:
            f = disparity_ncorr

        D_L = f(self._smoothed_imgL, self._smoothed_imgR, 1, self.w_size, self.dmax)
        # D_R = f(self._smoothed_imgR, self.imageL, 0, self.w_size, self.dmax)
        #
        # D_R = cv2.normalize(D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        self.imgDL = D_L
        # self.imgDR = D_R
        D_L = cv2.normalize(D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # cv2.imshow('disparityMap',disparity)
        cv2.imshow('L',self.imgDL)

