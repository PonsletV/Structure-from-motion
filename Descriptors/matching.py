import numpy as np
from utils.homography import RansacModel, make_homog, H_from_ransac


class Match(object):
    def __init__(self, detector, img_left, img_right):
        self.img_left = img_left
        self.img_right = img_right

        self.img_shape = img_left.shape

        self.detector = detector

        self.is_fitted = False
        self.kp_l = None
        self.kp_r = None
        self.matches = None
        self.H = None

    def fit(self):
        """ compute the keypoints and descriptors """
        self.kp_l, des_l = self.detector.detect(self.img_left)
        self.kp_r, des_r = self.detector.detect(self.img_right)
        self.matches = self.detector.match(des_l, des_r)
        self.is_fitted = True

    def select_matches(self, match_threshold=None):
        if match_threshold is None:
            match_threshold = 0.2*self.img_shape[0]
        if not self.is_fitted:
            raise ValueError("must fit Match object first")
        self.matches, self.H = select_matches(self.kp_l, self.kp_r, self.matches, match_threshold)


class MultiMatch(object):
    def __init__(self, detector, img_list):
        self.img_list = img_list
        self.img_shape = img_list[0].shape
        self.n_img = len(img_list)

        self.detector = detector

        self.is_fitted = False

        self.kp_list = []
        self.matches = []

    def fit(self, match_threshold=None):
        """ We match each image with the following one """

        # We will first compute the keypoints and descriptors for each images
        i = 0
        des_list = []
        for img in self.img_list:
            kp, des = self.detector.detect(img)
            self.kp_list.append(kp)
            des_list.append(des)

            i += len(kp)

        if match_threshold is None:
            match_threshold = 0.1*self.img_shape[0]

        matches = []

        # Here we match each image with the following

        for i in range(self.n_img-1):
            match, _ = select_matches(self.kp_list[i], self.kp_list[i+1],
                                      self.detector.match(des_list[i], des_list[i+1]), match_threshold)

            matches.append(match)

        self.matches = matches
        self.is_fitted = True

    def get_match(self, i):
        """ return a "Match" object between the i and i+1 image """
        match = Match(self.detector, self.img_list[i], self.img_list[i+1])
        match.is_fitted = True
        match.kp_l = self.kp_list[i]
        match.kp_r = self.kp_list[i+1]
        match.matches = self.matches[i]
        return match


def select_matches(kp1, kp2, matches, match_threshold=100):
    """Select robust matches using the RANSAC algorithm"""

    model = RansacModel()

    pts1 = make_homog(np.transpose(np.array([kp1[m.queryIdx].pt for m in matches])))
    pts2 = make_homog(np.transpose(np.array([kp2[m.trainIdx].pt for m in matches])))

    H, inliers = H_from_ransac(pts1, pts2, model, maxiter=1000, match_threshold=match_threshold)
    inliers = np.sort(inliers)

    return matches[inliers], H
