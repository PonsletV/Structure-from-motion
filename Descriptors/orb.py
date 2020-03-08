import cv2
import numpy as np


class Orb(object):

    def __init__(self, nfeatures=10000, scaleFactor=1.3, nlevels=10, edgeThreshold=31, firstLevel=0, WTA_K=2,
                patchSize=31, fastThreshold=20):
        self.detector = cv2.ORB_create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K,
                                       patchSize, fastThreshold)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    def detect(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        return np.array(kp), np.array(des)

    def match(self, des1, des2):

        matches = self.matcher.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.queryIdx)

        return np.array(matches)
