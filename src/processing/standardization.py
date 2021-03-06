import cv2 as cv
import numpy as np
from ..other.other import *
from ..processing.mask_and_contour import get_mask


def invers_color(image):
    """

    Args:
        image: grayscale fingeprint image
    Function inverses color if mean color more than Gmax/2.
    Where Gmax is max grayscale value.

    Returns: invers grayscale image

    """
    if image.mean() > 127.5:
        inv_image = cv.bitwise_not(image)
    else:
        inv_image = image.copy()

    return inv_image


def get_proportion(im):
    """

    Args:
        im: image
    Function for getting proportion of image

    Returns: ration of height and width image

    """
    propotion = im.shape[0] / im.shape[1]
    return propotion


def get_golden_ratio_image(im):
    """
    Args:
        im: 2d ndarray
            image
    Returns: 2d ndarray
        Function returns image with image.shape[1]/image.shape[0]
    """
    phi = (np.sqrt(5) + 1) / 2
    pr = get_proportion(im)

    if pr > phi:
        new_height = int(im.shape[1] * phi)
        new_im = im[:new_height, :]

        return new_im
    else:
        return im


def crop_fp_image(fp_image, bound_const=5, with_golden_ratio=True):
    """

    Args:
        fp_image: fingerprint image
        bound_const: width of border around fingerprint image
        with_golden_ratio: bool. Use or not golden ratio proportion
    Function use convex hull to find finger and crop it

    Returns: fingerprint image without background

    """
    # invert colors (black -> white, white -> black)
    # and binarize image

    temp_image = fp_image

    # get contours and unite contours
    res = cv.findContours(image=temp_image, mode=cv.RETR_EXTERNAL,
                          method=cv.CHAIN_APPROX_SIMPLE)

    contours, hierarchy = find_contours_result(res)
    new_contour = np.vstack(contours)

    # find parameters of bounding rectangle
    x1, y1, width, height = cv.boundingRect(new_contour)
    x2, y2 = x1 + width, y1 + height
    x1, y1 = x1, y1

    # check coordinates of rectangle
    if x1 < 0:
        x1 = 0

    if y1 < 0:
        y1 = 0

    if x2 > temp_image.shape[1]:
        x2 = temp_image.shape[1]

    if y2 > temp_image.shape[0]:
        y2 = temp_image.shape[0]

    # make new image
    resized_image = temp_image[y1:y2, x1:x2]

    # crop bottom peace of fingeprint
    if with_golden_ratio:
        resized_image = get_golden_ratio_image(resized_image)

    resized_image = cv.copyMakeBorder(resized_image, bound_const, bound_const,
                      bound_const, bound_const, cv.BORDER_CONSTANT, value=0)
    return resized_image


def resize_fp_image(im, new_width=400, interpolation=cv.INTER_LINEAR):
    """

    Args:
        im: fingerprint image
        new_width: new fingerprint image width
        interpolation: interpolation methods

    Function resizes image to new image with new_width and height with saving proportion

    Returns: resized image with new_width

    """
    height, width = im.shape
    k = height/width

    resized_image = cv.resize(im, (new_width, int(new_width*k)), interpolation=interpolation)

    return resized_image


def standardize_image(im, bound_const=5, new_width=400, interpolation=cv.INTER_LINEAR,
                      with_golden_ratio=True, with_ench=True, ker_ksize=20, p0=0.1, p1=.45):
    """

    Args:
        im: fingerprint imgae
        bound_const: width of border around fingerprint image
        new_width: new_width of fingerprint image
        interpolation: Interpolation method
        with_golden_ratio: bool. Use or not golden ratio proportion
        with_ench: use enhancement
        ker_ksize: disk kernel size
        p0: first parameter
        p1: second parameter

    Function for getting standardize fingerprint image

    Returns: standardize fingerprint image
    """
    
    inv_im = invers_color(im)
    if with_ench:
        mask = get_mask(inv_im, ker_ksize=ker_ksize, p0=p0, p1=p1) == 255
        inv_im[mask == False] = 0

    crop_im = crop_fp_image(inv_im)
    resized_im = resize_fp_image(crop_im)
    
    return resized_im


def parse_args():
    import argparse

    ap = argparse.ArgumentParser()

if __name__ == '__main__':
    pass ## ?????