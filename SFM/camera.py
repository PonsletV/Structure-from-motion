from scipy import linalg
import numpy as np
import cv2


class Camera(object):
    """ Class for representing pin-hole cameras. """

    def __init__(self, Proj, calib):
        """ Initialize P = K[R|t] camera model. """
        self.Projection = Proj
        self.Calibration = calib # calibration matrix
        self.c = None

    def project(self, X):
        """ Project points in X (3*n array) """
        return cv2.projectPoints(X, self.Projection.rv, self.Projection.t,
                                 self.Calibration.K, distCoeffs=self.Calibration.dist_coeff)

    def center(self):
        """ Compute and return the camera center. """

        if self.c is not None:
            return self.c
        else:
            # compute c by factoring
            self.c = -np.dot(self.Projection.R.T, self.Projection.t)
            return self.c


class Projection(object):
    """ Class for representing P = [R|t]"""
    def __init__(self, P = np.hstack((np.eye(3), np.zeros((3, 1))))):
        if P is not None:
            self.P = P
            self.R = None
            self.t = None
            self.rv = None
            self.factor()

    def __getitem__(self, item):
        return self.P[item]

    def factor(self):
        self.R = self.P[:, :3]
        self.t = np.array([[self[0, 3]], [self[1, 3]], [self[2, 3]]])
        self.rv, _ = cv2.Rodrigues(self.R)


class Calibration(object):
    """ class for storing K and dist_coeff camera and compute focal"""
    def __init__(self, K, dist_coeff):
        self.K = K
        self.dist_coeff = dist_coeff

    def focal(self):
        """ Compute the mean focal """
        return (self.K[0, 0] + self.K[1, 1]) / 2


def rotation_matrix(a):
    """ Creates a 3D rotation matrix for rotation aroud the axis of the vector a. """
    R = np.eye(4)
    R[:3,:3] = linalg.expm([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return R


def cal(sz, fx, fy):
    row, col = sz
    K = np.diag([fx, fy, 1])
    K[0, 2] = 0.5*col
    K[1, 2] = 0.5*row
    return K


def concat(R, t):
    return Projection(np.hstack((R, t)))


def compose(P1 : Projection, P2 : Projection):
    t1 = P1.t
    t2 = P2.t
    R1 = P1.R
    R2 = P2.R
    t_1p2 = np.array([[t1[0][0] + t2[0][0]], [t1[1][0] + t2[1][0]], [t1[2][0] + t2[2][0]]])

    R_1p2 = np.dot(R1, R2)
    return concat(R_1p2, t_1p2)
