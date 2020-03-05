import numpy as np
from mpl_toolkits.mplot3d import axes3d
from utils.homography import make_homog
from matplotlib import pyplot as plt
from SFM.SFM import Sfm
from Descriptors.matching import Match


def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.

    return np.concatenate((im1, im2), axis=1)


def plot_matches(match: Match):
    pts1 = make_homog(np.transpose(np.array([match.kp_l[m.queryIdx].pt for m in match.matches])))
    pts2 = make_homog(np.transpose(np.array([match.kp_r[m.trainIdx].pt for m in match.matches])))

    img3 = appendimages(match.img_left, match.img_right)

    plt.imshow(img3)
    plt.gray()

    cols1 = match.img_left.shape[1]

    for i in range(len(match.matches)):
        plt.plot([pts1[0][i], pts2[0][i] + cols1], [pts1[1][i], pts2[1][i]], 'c', linewidth=0.5, linestyle='--')
        plt.axis('off')

    plt.plot(pts1[0], pts1[1], 'r.')
    plt.axis('off')

    plt.plot(pts2[0] + cols1, pts2[1], 'r.')
    plt.axis('off')

    plt.show()


def plot_3D_points(points3D, P_list=None):
    """
    Print the 3D points and the camera positions
    :param points3D: array of 3D points
    :param P_list: list of projections or projection matrix [R|t]
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(points3D.T[0], points3D.T[1], points3D.T[2], 'k.')
    if P_list is not None:
        for P in P_list:
            ax.plot([P[0, 3]], [P[1, 3]], [P[2, 3]], 'r.')
    plt.show()


def plot_reprojection(sfm : Sfm, img_idx="all"):
    """
    Print the points of the sfm and their reprojection
    :param sfm: Structure from motion object
    :param img_idx: "l" or "r" to plot left or right image
    """

    if img_idx == "l":
        im = sfm.img_left
        xp = sfm.xp_l
        x = sfm.x_l
    elif img_idx == "r":
        im = sfm.img_right
        xp = sfm.xp_r
        x = sfm.x_r
    else:
        plot_reprojection(sfm, "l")
        plot_reprojection(sfm, "r")
        return None

    plt.figure()
    plt.imshow(im)
    plt.gray()
    plt.plot(xp[0, :], xp[1, :], 'o')
    plt.plot(x[0, :], x[1, :], 'r.')
    plt.axis('off')
