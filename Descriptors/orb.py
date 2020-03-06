import cv2
import numpy as np


class Orb(object):

    def __init__(self, nfeatures=10000, scaleFactor=1.3, nlevels=10, edgeThreshold=31, firstLevel=0, WTA_K=2,
                patchSize=31, fastThreshold=20):
        self.detector = cv2.ORB_create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K,
                                       patchSize, fastThreshold)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    def fit(self, img1, img2):
        kp1, kp2 = self.detector.detect(img1), self.detector.detect(img2)
        kp1, des1 = self.detector.compute(img1, kp1)
        kp2, des2 = self.detector.compute(img2, kp2)

        matches = self.matcher.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        return np.array(kp1), np.array(kp2), np.array(des1), np.array(des2), np.array(matches)
