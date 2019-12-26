import cv2 as cv

from skimage.filters.rank import enhance_contrast_percentile
from skimage.morphology import disk
from .other import find_contours_result


def threshold_with_enhance(im, ker_ksize=20, p0=0.1, p1=.6):
    disk_ker = disk(ker_ksize)
    ecnh_im = enhance_contrast_percentile(inv_im, disk_ker, p0=p0, p1=p1)
    thresh = cv.threshold(ecnh_im, 0, 255, cv.THRESH_BINARY)[1]
    
    return thresh


def get_mask(im, ker_ksize=20, p0=0.1, p1=.6):
    thresh_im = threshold_with_enhance(im, ker_ksize=ker_ksize, p0=p0, p1=p1)
    res = cv.findContours(thresh_im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contours, hierarchy = find_contours_result(res)
    merge_contours = np.vstack(contours)

    hull = cv.convexHull(merge_contours)

    mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    mask = cv.fillConvexPoly(mask, hull, 255)

    return mask