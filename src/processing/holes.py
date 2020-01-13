import numpy as np
from .mask_and_contour import get_mask
import cv2 as cv


def get_horizontal_vertical_mean_percentiles(im, percs=[30, 60, 90, 100]):
    result = np.array([])
    mask = get_mask(im, with_ench=False)

    inv_im = cv.bitwise_not(im)
    inv_im[mask == False] = 0

    orig_v_sum = np.percentile(im.mean(axis=0), percs)
    orig_h_sum = np.percentile(im.mean(axis=1), percs)
    inv_v_sum = np.percentile(inv_im.mean(axis=0), percs)
    inv_h_sum = np.percentile(inv_im.mean(axis=1), percs)
    result = np.append(result, [orig_h_sum, orig_v_sum, inv_h_sum, inv_v_sum])

    return result

