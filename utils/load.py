import glob
import cv2


def load(dirpath):
    images = glob.glob(dirpath+'/*.jpg')
    im_list = []
    for fname in images:
        img = cv2.imread(fname, 0)
        im_list.append(img)
    return im_list