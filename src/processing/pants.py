import cv2 as cv
import numpy as np
from .border import *
from ..other.other import find_contours_result

def r_max(image, flag='h', percs=[80, 90, 100]):
    
    res = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = find_contours_result(res)
    
    radiuses = []
    for cont in contours:
        p, r = cv.minEnclosingCircle(cont)
        radiuses.append(r)

    if flag == 'h':
        return np.percentile(radiuses, percs) / image.shape[0]
    elif flag == 'v':
        return np.percentile(radiuses, percs) / image.shape[1]
    else:
        return None


def distance_transform_percentiles(image, percs=[90, 95, 100]):
    ds = cv.distanceTransform(image, cv.DIST_L1, cv.DIST_MASK_3)
    ds_ = ds[ds != 0]

    return np.percentile(ds_, percs)


def r_max_and_ds_percentiles(image, r_percs=[90, 95, 100], ds_percs=[90, 95, 100]):
    left, right = left_right_border(image)
    top, _ = top_bottom_border(image)

    rs_left = r_max(left, flag='h', percs=r_percs)
    rs_right = r_max(right, flag='h', percs=r_percs)
    rs_top = r_max(top, flag='v', percs=r_percs)

    ds_left = distance_transform_percentiles(left, percs=ds_percs)
    ds_right = distance_transform_percentiles(right, percs=ds_percs)
    ds_top = distance_transform_percentiles(top, percs=ds_percs)

    result = np.array([])
    result = np.append(result, [rs_left, rs_right, rs_top, ds_left, ds_right, ds_top])

    return result