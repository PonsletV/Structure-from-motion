from scipy import linalg
import numpy as np
import cv2


class Camera(object):
    """ Class for representing pin-hole cameras. """

    def __init__(self, projection, calibration):
        """ Initialize P = K[R|t] camera model. """
        self.projection = projection
        self.calibration = calibration # calibration matrix
        self.c = None

    def project(self, X):
        """ Project points in X (3*n array) """
        x = cv2.projectPoints(X, self.projection.rv, self.projection.t,
                              self.calibration.K, distCoeffs=self.calibration.dist_coeff)[0]
        return np.vstack((x[:, 0, 0], x[:, 0, 1]))

    def center(self):
        """ Compute and return the camera center. """

        if self.c is not None:
            return self.c
        else:
            # compute c by factoring
            self.c = -np.dot(self.projection.R.T, self.projection.t)
            return self.c

    def to_bundle(self):
        rv = self.projection.rv
        t = self.projection.t
        #f = self.calibration.focal()
        #k1 = self.calibration.dist_coeff[:, 0]
        #k2 = self.calibration.dist_coeff[:, 1]
        return np.array([rv[0][0], rv[1][0], rv[2][0],
                         t[0][0], t[1][0], t[2][0]], dtype=float)


class Projection(object):
    """ Class for representing P = [R|t]"""
    def __init__(self, P = np.hstack((np.eye(3), np.zeros((3, 1))))):
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
    R[:3, :3] = linalg.expm([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
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


def camera_from_bundle(cameraArray, K: Calibration):
    rv = cameraArray[:3]
    t = cameraArray[3:]
    t_array = np.array([[t[0]], [t[1]], [t[2]]])

    R = cv2.Rodrigues(rv)[0]
    P = concat(R, t_array)
    return Camera(P, K)
