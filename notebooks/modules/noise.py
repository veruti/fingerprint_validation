import cv2 as cv
import numpy as np
from .mask_and_contour import get_mask
from itertools import combinations as comb

def get_gradient(image, rx=1, ry=1):

    dx = cv.Sobel(image, cv.CV_64F, rx, 0)
    dy = cv.Sobel(image, cv.CV_64F, 0, ry)

    return dx, dy


def avg_grad(gx, gy, ksize, with_sqrt=True):
    gsq_real = np.square(gx) - np.square(gy)
    gsq_imag = 2 * gx * gy

    gsq_avg_real = cv.GaussianBlur(gsq_real, (ksize,) * 2, -1)
    gsq_avg_imag = cv.GaussianBlur(gsq_imag, (ksize,) * 2, -1)

    if with_sqrt:
        g_avg = np.sqrt(gsq_avg_real + np.complex(0, 1) * gsq_avg_imag)

        real_part = g_avg.real
        complex_part = g_avg.imag
        return real_part, complex_part
    else:
        return gsq_avg_real, gsq_avg_imag


def norm_vector(dx, dy, eps=1e-3):
    lengths = np.sqrt(np.square(dx) + np.square(dy))
    norm_dx, norm_dy = np.divide(dx, np.add(lengths, eps)), np.divide(dy, np.add(lengths, eps))

    return norm_dx, norm_dy


def scalar_product(dx, dy, ksize=7):
    part1 = cv.GaussianBlur(dx, (ksize,) * 2, -1) * dx
    part2 = cv.GaussianBlur(dy, (ksize,) * 2, -1) * dy

    sp = part1 + part2

    return sp


def get_scalar_product_precentiles(im, ksize=13, percs=[0, 20, 40, 60]):
    dx, dy = get_gradient(im)
    dx_avg, dy_avg = avg_grad(dx, dy, ksize=ksize, with_sqrt=False)
    dx_norm, dy_norm = norm_vector(dx_avg, dy_avg)

    sp = scalar_product(dx_norm, dy_norm, ksize=ksize)
    mask = get_mask(im, with_ench=False) == 255

    percentiles = np.percentile(sp[mask], percs)

    return percentiles


def get_pair_scalar_product_distribution(im, ksize=(17, 57)):
    dx, dy = get_gradient(im)

    dx_avg1, dy_avg1 = avg_grad(dx, dy, ksize[0])
    dx_avg2, dy_avg2 = avg_grad(dx, dy, ksize[1])

    mask0 = get_mask(im, with_ench=False) == 255
    mask1 = np.sqrt(dx_avg1 * dx_avg1 + dy_avg1 * dy_avg1) > 0.5
    mask2 = np.sqrt(dx_avg2 * dx_avg2 + dy_avg2 * dy_avg2) > 0.5

    mask = mask0 & mask1 & mask2

    dx_avg1, dy_avg1 = norm_vector(dx_avg1, dy_avg1)
    dx_avg2, dy_avg2 = norm_vector(dx_avg2, dy_avg2)

    sp = dx_avg1[mask] * dx_avg2[mask] + dy_avg1[mask] * dy_avg2[mask]

    return sp


def get_pair_scalar_product_percentiles(im, ksize=(17, 65), ps=[5, 10, 20, 50]):
    pair_scalar_product = get_pair_scalar_product_distribution(im, ksize=ksize)
    percentiles = np.percentile(pair_scalar_product, ps)

    return percentiles


def get_all_pair_sp_percentiles(im, ksizes=(17, 33, 65), ps=[5, 10, 20, 50]):
    sp = np.array([])
    for ksize in comb(ksizes, 2):
        sp_ = get_pair_scalar_product_percentiles(im, ksize=ksize,ps=ps)
        sp = np.append(sp, sp_)

    return sp

