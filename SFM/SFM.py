import numpy as np
import cv2
from Descriptors.matching import Match
from SFM.camera import Projection, compose, concat, Calibration


class Sfm(object):
    """ Class for representing the correspondence between two images"""
    def __init__(self, calibration: Calibration, match: Match, projection_left: Projection):
        """
        Initialize the structure needed to reconstruct the 3D points from the scene
        :param calibration: calibration object (contains the K matrix)
        :param match: Match object with the point detector and images
        :param projection_left: projection object, current position of the left camera
        """
        self.calibration = calibration
        self.match = match
        self.P_left = projection_left
        self.img_shape = match.img_shape

        self.is_fitted = False

        """ reconstruction properties """
        self.P_transformation = None
        self.X = None
        self.E = None
        self.F = None
        self.P_right = None

        """ points """
        self.x_l = None
        self.x_r = None
        self.xp_l = None
        self.xp_r = None
        self.threeDpoints = None

    def fit_descriptor(self):
        """ compute the keypoints and descriptors, lowe_factor only used for SIFT"""
        self.match.fit()

    def fit_reconstruction(self):
        """
        Reconstruct the 3D points (after fit_descriptor)
        :return: list of 3D points of size (3*n)
        """
        if not self.match.is_fitted:
            raise ValueError("must call fit_descriptor before")

        K = self.calibration.K
        P_l = self.P_left

        # get the coordinates of the points from the keypoints lists
        pts_l = np.array([self.match.kp_l[m.queryIdx].pt for m in self.match.matches])
        pts_r = np.array([self.match.kp_r[m.trainIdx].pt for m in self.match.matches])

        # estimate E with RANSAC
        E, mask = cv2.findEssentialMat(pts_l, pts_r, cameraMatrix=self.calibration.K, method=cv2.FM_RANSAC)
        inliers = np.where(mask == 1)[0]

        # select the inliers from RANSAC
        pts_l = pts_l[inliers, :]
        pts_r = pts_r[inliers, :]

        # compute the fundamental F
        self.F = np.dot(np.linalg.inv(K.T), np.dot(E, np.linalg.inv(K)))

        # compute the rotation and translation of the transformation
        _, R, t, mask_c = cv2.recoverPose(E, pts_l, pts_r, cameraMatrix=K)
        self.P_transformation = Projection(concat(R, t))

        # compute the new position of the camera
        P_r = compose(P_l, concat(R, t))

        # select the points that are in front of the selected camera
        infront = np.where(mask_c != 0)[0]
        pts_l = pts_l[infront, :]
        pts_r = pts_r[infront, :]

        # find the position of the 3D points with a triangulation
        X = cv2.triangulatePoints(np.dot(K, P_l.P), np.dot(K, P_r.P), pts_l.T, pts_r.T)
        X = X / X[3]
        threeDpoints = X[:3, :]

        # reproject the 3D points through the cameras
        xlp, _ = cv2.projectPoints(threeDpoints.T, P_l.rv, P_l.t, K, distCoeffs=self.calibration.dist_coeff)
        xrp, _ = cv2.projectPoints(threeDpoints.T, P_r.rv, P_r.t, K, distCoeffs=self.calibration.dist_coeff)

        # kept points from the initial keypoints on the two images
        self.x_l = pts_l.T
        self.x_r = pts_r.T

        # reprojected points
        self.xp_l = np.vstack((xlp[:, 0, 0], xlp[:, 0, 1]))
        self.xp_r = np.vstack((xrp[:, 0, 0], xrp[:, 0, 1]))

        # save the results
        self.E = E
        self.P_right = P_r
        self.threeDpoints = threeDpoints
        self.is_fitted = True

        return threeDpoints

    def fit(self):
        """ compute the matchings between the two images then the 3D points corresponding to the keypoints"""
        self.fit_descriptor()
        return self.fit_reconstruction()

    def compute_reprojection_error(self, verbose=False):
        if not self.is_fitted:
            raise ValueError("must be fitted before computation")

        error1 = np.linalg.norm(self.x_l - self.xp_l, axis=0).mean()
        error2 = np.linalg.norm(self.x_r - self.xp_r, axis=0).mean()

        if verbose:
            print('Reprojection error on left image : ', error1)
            print('Reprojection error on right image : ', error2)
        return error1, error2

    def num_points(self):
        if self.threeDpoints is not None:
            return self.threeDpoints.shape[0]
        else:
            return 0
