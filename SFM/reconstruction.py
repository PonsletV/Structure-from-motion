import numpy as np
import networkx as nx
import cv2
from Descriptors.matching import Match, MultiMatch
from SFM.camera import Projection, compose, concat, Calibration, Camera, camera_from_bundle
from SFM.SFM import Sfm
from utils.PySBA import PySBA


class Reconstruction(object):
    """

    """

    def __init__(self, match: MultiMatch, calibration: Calibration):
        """
        Initialize the pipeline used for reconstruction
        :param match: a MultiMatch object
        :param calibration:
        """

        self.match = match
        self.calibration = calibration

        self.n_reconstruction = len(self.match.matches)

        self.sfm_list = []

        self.camera_list = []
        self.points2D = None
        self.camera_ind = None
        self.points_ind = None

    def reconstruct(self):
        projection_left = Projection()  # By default we use no rotation nor translation for the first camera
        self.camera_list.append(Camera(projection_left, self.calibration))

        X = np.array([], dtype=np.float).reshape(3, 0)

        count_3D = 0

        twoD_points = np.array([], dtype=np.float).reshape(2, 0)
        kp_to_points = []
        kp_view = []

        for (i, kpl) in enumerate(self.match.kp_list):
            kp_to_points += [[] for _ in kpl]
            kp_view += [i]*len(kpl)
            twoD_points = np.hstack((twoD_points, np.array([kp.pt for kp in kpl]).T))

        index_i = 0
        index_ipp = 0

        for i in range(self.n_reconstruction):
            sfm = Sfm(self.calibration, self.match.get_match(i), projection_left)
            X = np.hstack((X, sfm.fit_reconstruction()))

            self.sfm_list.append(sfm)

            self.camera_list.append(Camera(sfm.P_right, self.calibration))
            self.match.matches[i] = self.match.matches[i][sfm.mask]

            projection_left = sfm.P_right

            index_ipp += len(self.match.kp_list[i])

            for i_l, i_r in zip(sfm.idx_l, sfm.idx_r):  # for each point in the new image
                kp_to_points[i_l+index_i].append(count_3D)
                kp_to_points[i_r+index_ipp].append(count_3D)
                count_3D += 1

            index_i = index_ipp

        points_3d = self.postprocessing(kp_to_points, twoD_points, kp_view, X)

        cameraArray = np.array([cam.to_bundle() for cam in self.camera_list])

        bundle_adjustment = PySBA(cameraArray, points_3d.T, self.points2D.T, self.camera_ind, self.points_ind, self.calibration)
        cameras, points3D = bundle_adjustment.bundleAdjust()
        self.points3D = points3D.T

        c_list = []
        for i in range(cameras.shape[0]):
            c_list.append(camera_from_bundle(cameraArray[i], self.calibration))
        self.camera_list = c_list

        print("Mean-squared Error before B-A: ", self.MSE(points_3d))
        print("Mean-squared Error after B-A: ", self.MSE(self.points3D))
        # return self.points3D
        return points_3d

    def MSE(self, points3D):
        n_images = self.match.n_img
        error = []
        for i in range(n_images):
            c = self.camera_list[i]
            index = np.where(self.camera_ind == i)[0]
            xp = c.project(points3D.T[self.points_ind[index]])
            x = self.points2D[:, index]
            error.append(np.mean(np.linalg.norm(xp - x, axis=0)))
        return error

    def postprocessing(self, kp_to_points, twoD_points, kp_view, X):
        G = nx.Graph()
        points_2d = np.array([], dtype=np.float).reshape(2, 0)
        camera_ind = []
        points_ind = []
        points_3d = np.array([], dtype=np.float).reshape(3, 0)
        count_3d = 0
        already_counted = []

        for (i, kp) in enumerate(kp_to_points):
            if len(kp) == 1:
                points_2d = np.column_stack((points_2d, twoD_points[:, i]))
                camera_ind.append(kp_view[i])
                if kp[0] not in already_counted:
                    points_3d = np.column_stack((points_3d, X[:, kp[0]]))
                    points_ind.append(count_3d)
                    count_3d += 1
                    already_counted.append(kp[0])
                else:
                    points_ind.append(already_counted.index(kp[0]))
            elif len(kp) == 2:
                G.add_edge(kp[0], kp[1], weight=i)

        conn = nx.connected_components(G)
        for g in conn:  #those 3d points all represents the same object point
            kpl = []  # List of keypoint indices related to the points of g
            for u in g:
                for v in g:
                    if u < v:
                        a = G.get_edge_data(u, v)
                        if a is not None:
                            kpl.append(a['weight'])

            x = np.zeros(3)
            for idx in g:
                x += X[:, idx]
            x = x / len(g)

            for kp_idx in kpl:
                points_2d = np.column_stack((points_2d, twoD_points[:, kp_idx]))
                camera_ind.append(kp_view[kp_idx])
                points_ind.append(count_3d)
            points_3d = np.column_stack((points_3d, x))
            count_3d += 1

        self.points2D = points_2d
        self.camera_ind = np.array(camera_ind, dtype=np.int16)
        self.points_ind = np.array(points_ind, dtype=np.int32)

        return points_3d
