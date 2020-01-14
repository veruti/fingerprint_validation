import cv2 as cv
import numpy as np
from skimage.filters.rank import enhance_contrast_percentile
from skimage.morphology import disk
from ..other.other import find_contours_result


def threshold_with_enhance(im, ker_ksize=20, p0=0.1, p1=.45):
    """

    Args:
        im: fingerprint image
        ker_ksize: disk kernel size
        p0: first parameter
        p1: second parameter

    Use  enhance contrast percentile for getting binary inv
    Returns: binary fingeprint image

    """
    disk_ker = disk(ker_ksize)
    enh_im = enhance_contrast_percentile(im, disk_ker, p0=p0, p1=p1)
    thresh = cv.threshold(enh_im, 0, 255, cv.THRESH_BINARY)[1]
    
    return thresh


def get_mask(im, ker_ksize=20, p0=0.1, p1=.45, with_ench=True):
    """

    Args:
        with_ench: bool
            use image enhance
        im: ndarray
            fingeprint image
        ker_ksize: int
            disk kernel size
        p0: float [0,...,1] p0 < p1
            first parameter
        p1: float [0,...,1] p0 < p1
            second parameter

    Function for getting mask of fingerprint image

    Returns: mask of fingerprint image

    """
    if with_ench:
        thresh_im = threshold_with_enhance(im, ker_ksize=ker_ksize, p0=p0, p1=p1)
        res = cv.findContours(thresh_im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    else:
        res = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contours, hierarchy = find_contours_result(res)
    merge_contours = np.vstack(contours) 
    
    hull = cv.convexHull(merge_contours)
    
    mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    mask = cv.fillConvexPoly(mask, hull, 255)
    
    return mask