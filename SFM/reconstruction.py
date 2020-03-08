import numpy as np
import networkx as nx
import cv2
from Descriptors.matching import Match, MultiMatch
from SFM.camera import Projection, compose, concat, Calibration, Camera
from SFM.SFM import Sfm
from utils.PySBA import PySBA


class Reconstruction(object):
    """

    """

    def __init__(self, match: MultiMatch, calibration: Calibration):
        """

        :param bundle_adjustment:
        :param match:
        """

        self.match = match
        self.calibration = calibration

        self.n_reconstruction = len(self.match.matches)

        self.camera_list = []
        self.points2D = None
        self.camera_ind = None
        self.points_ind = None

    def reconstruct(self):
        projection_left = Projection()  # By default we use no rotation nor translation for the first camera
        self.camera_list.append(Camera(projection_left, self.calibration))

        X = np.array([], dtype=np.float).reshape(3, 0)
        G = nx.Graph()
        space_i = 0
        space_ipp = 0
        points2D = np.array([], dtype=np.float).reshape(2, 0)

        ind_points = []
        camera_ind = []

        last_idx_r = None
        count = 0
        for i in range(self.n_reconstruction):
            sfm = Sfm(self.calibration, self.match.get_match(i), projection_left)
            X = np.hstack((X, sfm.fit_reconstruction()))
            n_points = sfm.threeDpoints.shape[1]

            self.camera_list.append(Camera(sfm.P_right, self.calibration))
            self.match.matches[i] = self.match.matches[i][sfm.mask]

            points2D = np.hstack((points2D, np.hstack((sfm.x_l, sfm.x_r))))
            camera_ind += [i]*sfm.x_l.shape[1]
            camera_ind += [i+1]*sfm.x_r.shape[1]
            ind_points += list(zip(list(range(count, count + n_points)),
                                   list(range(count + sfm.x_r.shape[1], count + sfm.x_r.shape[1] + n_points))))
            count += 2*n_points
            projection_left = sfm.P_right

            # We link together indices of X which represents the same object point
            if last_idx_r is not None:
                space_ipp += last_idx_r.shape[0]
                for r, kp_idx in enumerate(last_idx_r):
                    if kp_idx in sfm.idx_l:
                        l = np.searchsorted(sfm.idx_l, kp_idx)
                        G.add_edge(r + space_i, l + space_ipp)
                space_i = space_ipp

            last_idx_r = sfm.idx_r

        points_ind = np.zeros(points2D.shape[1])

        # Now we will keep only one representation for points that appears in more than two views
        X_new = np.array([], dtype=np.float).reshape(3, 0)
        to_remove = []
        conn = nx.connected_components(G)
        i=0
        for g in conn:
            i += 1
            x = np.zeros(3)
            for idx in g:
                x += X[:, idx]
                to_remove.append(idx)
                (k, l) = ind_points[idx]
                points_ind[k] = i
                points_ind[l] = i
            x = x/len(g)
            X_new = np.column_stack((X_new, x))

        for idx in range(X.shape[1]):
            if idx not in to_remove:
                i += 1
                (k, l) = ind_points[idx]
                points_ind[k] = i
                points_ind[l] = i

        X = np.delete(X, np.array(to_remove), axis=1)
        X_new = np.hstack((X_new, X))

        # Now we will gather all the cameras and run the bundle adjustment
        self.points2D = points2D
        self.camera_ind = np.array(camera_ind, dtype=np.int16)
        self.points_ind = np.array(camera_ind, dtype=np.int32)

        cameraArray = np.array([cam.to_bundle() for cam in self.camera_list])
        bundle_adjustment = PySBA(cameraArray, X_new.T, points2D.T, self.camera_ind, self.points_ind)
        cameras, self.points3D = bundle_adjustment.bundleAdjust()

        return self.points3D