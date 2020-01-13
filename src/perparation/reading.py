import os
import cv2 as cv
from numpy.random import choice
from .filepaths import *


TYPES = ['.jpg', '.jpeg', '.jpe',
         '.png',
         '.bmp', '.dib',
         '.pbm', '.pgm', '.ppm',
         '.sr', '.ras',
         '.tiff', '.tif']


def check_im_type(name: str):
    """
    Args:
        name: image name
    Function for checking type of image

    Returns: bool
    True if image has one of TYPES type
    False if it's not an image
    """
    fl = False
    for tp in TYPES:
        if name.endswith(tp):
            fl = True
            break
    return fl


def images_dir_list(listdir):
    """
    Args:
        listdir: list of images paths

    Function for
    Returns: list of image paths

    """
    return [name for name in listdir if check_im_type(name)]


def read_random_image(file_path, with_path=False, flag=cv.IMREAD_GRAYSCALE):
    """

    Args:
        file_path: Str. File path to folder
        with_path: Bool. Return file_path
        flag: cv.IMREAD_* flag
    Read and return random image for file

    Returns: image

    """
    images_list = images_dir_list(os.listdir(file_path))
    image_name = choice(images_list)
    image_path = file_path + image_name

    if with_path:
        return image_path, cv.imread(image_path, flag)
    else:
        return cv.imread(image_path, flag)