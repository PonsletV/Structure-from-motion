from Descriptors.sift import Sift
from utils.load import load
from utils.calibrate import calibrate
from SFM.camera import Projection
from utils.display import *
from Descriptors.orb import Orb
#%%


im_list = load('dataset/sfm')
calibration = calibrate('dataset/calib')

detector = Orb(nfeatures=10000)
matcher = Match(detector, im_list[0], im_list[1])
matcher.fit()
matcher.select_matches(match_threshold=0.1*matcher.img_shape[0])

plot_matches(matcher)

#%%
initial_projection = Projection()
structure_from_motion = Sfm(calibration, matcher, initial_projection)
X = structure_from_motion.fit_reconstruction()

plot_reprojection(structure_from_motion)
plot_3D_points(X, [structure_from_motion.P_left, structure_from_motion.P_right])
