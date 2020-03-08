from Descriptors.sift import Sift
from Descriptors.matching import MultiMatch
from utils.load import load
from utils.calibrate import calibrate
from SFM.camera import Projection
from SFM.reconstruction import Reconstruction
from utils.display import *
from Descriptors.orb import Orb
from utils.PySBA import PySBA

im_list = load('dataset/sfm')
calibration = calibrate('dataset/calib')

detector = Sift(nfeatures=5000)
matcher = MultiMatch(detector, im_list)
matcher.fit()
reconstruction = Reconstruction(matcher, calibration)
X = reconstruction.reconstruct().T

plot_3D_points(X)
