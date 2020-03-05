import numpy as np
import cv2


class Sift(object):

    def __init__(self, nfeatures=5000, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, lowe_factor=.7):
        self.detector = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers,
                                                    contrastThreshold=contrastThreshold,
                                                    edgeThreshold=edgeThreshold, sigma=sigma)
        self.lowe_factor = lowe_factor

    def fit(self, img1, img2):

        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        bf = cv2.BFMatcher_create(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < self.lowe_factor * n.distance:
                good.append([m])

        return np.array(kp1), np.array(kp2), np.array(des1), np.array(des2), np.array(good)[:, 0]
