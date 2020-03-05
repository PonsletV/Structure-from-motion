import numpy as np
from utils.homography import RansacModel, make_homog, H_from_ransac


class Match(object):
    def __init__(self, detector, img_left, img_right):
        self.img_left = img_left
        self.img_right = img_right
        self.detector = detector

        self.is_fitted = False
        self.kp_l = None
        self.kp_r = None
        self.matches = None
        self.H = None

    def fit(self):
        """ compute the keypoints and descriptors """
        self.kp_l, self.kp_r, _, _, self.matches = self.detector.fit(self.img_left, self.img_right)
        self.is_fitted = True

    def select_matches(self, match_threshold=100):
        if not self.is_fitted:
            raise ValueError("must fit Match object first")
        self.matches, self.H = select_matches(self.kp_l, self.kp_r, self.matches, match_threshold)


def select_matches(kp1, kp2, matches, match_threshold=100):
    """Select robust matches using the RANSAC algorithm"""

    model = RansacModel()

    pts1 = make_homog(np.transpose(np.array([kp1[m.queryIdx].pt for m in matches])))
    pts2 = make_homog(np.transpose(np.array([kp2[m.trainIdx].pt for m in matches])))

    H, inliers = H_from_ransac(pts1, pts2, model, maxiter=1000, match_threshold=match_threshold)

    return matches[inliers], H
