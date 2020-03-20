from Descriptors.sift import Sift
from Descriptors.matching import MultiMatch
from utils.load import load
from utils.calibrate import calibrate
from SFM.camera import Projection
from SFM.reconstruction import Reconstruction
from utils.display import *
from Descriptors.orb import Orb


im_list = load('dataset/sfm/grillage')
calibration = calibrate('dataset/calib')

detector = Orb(nfeatures=5000)
matcher = MultiMatch(detector, im_list)
matcher.fit()


reconstruction = Reconstruction(matcher, calibration)
X_ba = reconstruction.reconstruct()
#for sfm in reconstruction.sfm_list:
#    plot_reprojection(sfm)

plot_reprojection_multi(reconstruction)
P_list = [cam.projection for cam in reconstruction.camera_list]
plot_3D_points(X_ba, P_list)

