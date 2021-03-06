import numpy as np
import cv2
from mpl_toolkits.mplot3d import axes3d
from utils.homography import make_homog
from matplotlib import pyplot as plt
from SFM.SFM import Sfm
from SFM.reconstruction import Reconstruction
from Descriptors.matching import Match, MultiMatch


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
    """ Plot the matches in object Match with lines showing the correspondances """
    pts1 = make_homog(np.transpose(np.array([match.kp_l[m.queryIdx].pt for m in match.matches])))
    pts2 = make_homog(np.transpose(np.array([match.kp_r[m.trainIdx].pt for m in match.matches])))

    img3 = appendimages(match.img_left, match.img_right)

    plt.imshow(img3)
    plt.gray()

    cols1 = match.img_shape[1]

    for i in range(len(match.matches)):
        plt.plot([pts1[0][i], pts2[0][i] + cols1], [pts1[1][i], pts2[1][i]], 'c', linewidth=0.5, linestyle='--')
        plt.axis('off')

    plt.plot(pts1[0], pts1[1], 'r.')
    plt.axis('off')

    plt.plot(pts2[0] + cols1, pts2[1], 'r.')
    plt.axis('off')

    plt.show()


def plot_kp(match: Match, img_ind="l"):
    if img_ind == "r":
        img = match.img_right
        kp = match.kp_r
    else:
        img = match.img_left
        kp = match.kp_l

    img2 = np.zeros_like(img)
    img2 = cv2.drawKeypoints(img, kp, color=(0, 255, 0), flags=0, outImage=img2)
    plt.imshow(img2)
    plt.show()


def plot_3D_points(points3D, P_list=None):
    """
    Print the 3D points and the camera positions
    :param points3D: array of 3D points
    :param P_list: list of projections or projection matrix [R|t]
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d([-5, 5])
    ax.set_ylim3d([-5, 5])
    ax.set_zlim3d([-5, 5])
    ax.plot(points3D[0], points3D[1], points3D[2], 'k.')
    if P_list is not None:
        for P in P_list:
            ax.plot([P[0, 3]], [P[1, 3]], [P[2, 3]], 'r.')
    plt.show()


def plot_reprojection(sfm: Sfm, img_idx="all"):
    """
    Print the points of the sfm and their reprojection
    :param sfm: Structure from motion object
    :param img_idx: "l" or "r" to plot left or right image
    """

    if img_idx == "l":
        im = sfm.match.img_left
        xp = sfm.xp_l
        x = sfm.x_l
    elif img_idx == "r":
        im = sfm.match.img_right
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
    plt.show()


def plot_reprojection_multi(reconstruction: Reconstruction):
    n_images = reconstruction.match.n_img
    for i in range(n_images):
        img = reconstruction.match.img_list[i]
        img_shape = reconstruction.match.img_shape

        c = reconstruction.camera_list[i]
        index = np.where(reconstruction.camera_ind == i)[0]
        xp = c.project(reconstruction.points3D.T[reconstruction.points_ind[index]])
        xp = xp[:, np.where(xp[0, :] <= img_shape[1])[0]]
        xp = xp[:, np.where(xp[0, :] >= 0)[0]]
        xp = xp[:, np.where(xp[1, :] <= img_shape[0])[0]]
        xp = xp[:, np.where(xp[1, :] >= 0)[0]]
        x = reconstruction.points2D[:, index]

        plt.figure()
        plt.imshow(img)
        plt.gray()
        plt.plot(xp[0, :], xp[1, :], 'o')
        plt.plot(x[0, :], x[1, :], 'r.')
        plt.axis('off')
