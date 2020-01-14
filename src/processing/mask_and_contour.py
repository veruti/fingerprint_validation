import cv2 as cv
import numpy as np
from skimage.filters.rank import enhance_contrast_percentile, entropy
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


def get_mask_enhance(im, ker_ksize=21, p0=0.1, p1=.45, with_ench=True):
    """
    Args:
        im: ndarray
            image
        ker_ksize: int
            kernel size
        p0: float [0,...,1] p0 < p1
            first percentile
        p1: float [0,...,1] p0 < p1
            second percentile
        with_ench: bool
            use enhance contrast percentile or not
    Function creates mask for fingerprint image

    Returns: ndarray
    return mask of fingerprint image
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


def threshold_with_entropy(im, ker_ksize=21, p=33):
    """
    Args:
        im: ndarray
            image
        ker_ksize: int
            kernel size
        p: int [0,...,100]
            lower percentiles
    Function creates mask of fingerprint image

    Returns: ndarray
    return mask of fingerprint image
    """
    disk_ker = disk(ker_ksize)
    im_entropy = entropy(im, disk_ker)

    perc0 = np.percentile(im_entropy, p)
    im_entropy[im_entropy < perc0] = 0

    thresh = cv.threshold(im_entropy, 0, 255, cv.THRESH_BINARY)[1]
    thresh = thresh.astype(np.uint8)

    return thresh


def get_mask_entropy(im, ker_ksize=20, p=33, with_ench=True):
    """
    Args:
        im: ndarray
            image
        ker_ksize: int
            kernel size
        p: int [0,...,100]
            lower percentiles
        with_ench: bool
            use entropy or not
    Function creates mask of fingerprint image

    Returns: ndarray
    Mask of fingerprint image

    """
    if with_ench:
        thresh_im = threshold_with_entropy(im, ker_ksize=ker_ksize, p=p)
        res = cv.findContours(thresh_im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    else:
        res = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contours, hierarchy = find_contours_result(res)
    merge_contours = np.vstack(contours)

    hull = cv.convexHull(merge_contours)

    mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    mask = cv.fillConvexPoly(mask, hull, 255)

    return mask


def get_mask(im, ker_ksize=21, p_entropy=33, with_ench=True, with_entropy=True, p0=0.1, p1=.45):
    """

    Args:
        im: ndarray
            image
        ker_ksize: int
            kernel size
        p_entropy: int [0,...,100]
            percentiles for entropy mask
        with_ench: bool
            use enchance(entropy) or not
        with_entropy: bool
            use entropy or enchance
        p0: float [0,...,1] p0<p1
            lower percentile of enchanche mask
        p1: float [0,...,1] p0<p1
            upper percentile of enchanche mask

    Returns:

    """
    if with_entropy:
        mask = get_mask_entropy(im, ker_ksize=ker_ksize, p=p_entropy, with_ench=with_ench)
    else:
        mask = get_mask_enhance(im, ker_ksize=ker_ksize, p0=p0, p1=p1, with_ench=with_ench)

    return mask