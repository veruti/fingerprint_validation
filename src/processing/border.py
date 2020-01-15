import numpy as np
import cv2 as cv


def left_right_border(img, left_perc=0.15, right_perc=0.15):
    """

    Args:
        img: ndarray
            image
        left_perc: float [0,...,1]
            left percentile
        right_perc: float [0,...,1]
            right percentile
    Get left and right peaces of image

    Returns: ndarray, ndarray
    Left and right peaces of image
    """
    cs = np.cumsum(np.sum(img, axis=0))
    cs = cs / cs[-1]
    j1 = np.argmax(cs > left_perc)
    j2 = cs.size - np.argmax(cs[::-1] < 1-right_perc)
    
    contour_left = img.copy()
    contour_right = img.copy()
    
    contour_left[:, j1:] = 0
    contour_right[:, :j2] = 0     
    
    return contour_left, contour_right


def top_bottom_border(img, top_perc=0.10, bottom_perc=0.10):
    """
    Args:
        img: ndarray
        top_perc: float [0,...,1]
            top percentile
        bottom_perc: float [0,...,1]
            bottom percentile
    Get left and right peaces of image

    Returns: ndarray, ndarray
    Top and bottom peaces of image

    """
    cs = np.cumsum(np.sum(img, axis=1))
    cs = cs / cs[-1]
    i1 = np.argmax(cs > top_perc)
    i2 = cs.size - np.argmax(cs[::-1] < 1-bottom_perc)
    
    contour_top = img.copy()
    contour_bottom = img.copy()
    
    contour_top[i1:, :] = 0
    contour_bottom[:i2, :] = 0   
    
    return contour_top, contour_bottom


def distance_transform_percentiles(image, percs=[90, 95, 100]):
    """
    Args:
        image: ndarray
            image
        percs: floats [0,...,1]
            Percentiles of distance transform distribution
    Function returns precentiles of Percentiles of distance transform distribution

    Returns: 1d ndarray
    Function returns
    """
    ds = cv.distanceTransform(image, cv.DIST_L1, cv.DIST_MASK_3)
    ds_ = ds[ds != 0]

    return np.percentile(ds_, percs)