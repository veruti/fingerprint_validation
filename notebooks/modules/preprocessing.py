import cv2 as cv
import numpy as np


def get_second_derivative(im, ksize=3):
    """

    Args:
        im: image
        ksize: odd int
            kernel size
    Function for getting second derivatives

    Returns: ndarray, ndarray, ndarray
    Function returns dxdx, dxdy and dydy derivatives
    """
    gxx = cv.Sobel(im, cv.CV_64F, 2, 0, ksize=ksize)
    gxy = cv.Sobel(im, cv.CV_64F, 1, 1, ksize=ksize)
    gyy = cv.Sobel(im, cv.CV_64F, 0, 2, ksize=ksize)
    
    return gxx, gxy, gyy


def coherence_filter(im, sigma = 15, str_sigma=15, blend=0.5, n_iter=5):
    """

    Args:
        im: image
        sigma: odd int
            kernel size for derivatives
        str_sigma: odd int
            kernel size for Eigen values and vectors
        blend: float [0, ... ,1]
            blend coefficient
        n_iter: int
            count of iterations
    Function makes coherence filter on image

    Returns: ndarray
    Function returns filtered image
    """
    h, w = im.shape[:2]

    for i in range(n_iter):
        eigen = cv.cornerEigenValsAndVecs(im, str_sigma, 3)
        eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
        
        x, y = eigen[:,:,1,0], eigen[:, :, 1, 1]

        gxx, gxy, gyy = get_second_derivative(im, sigma)
        gvv = x*x*gxx + 2*x*y*gxy + y*y*gyy
        
        m = gvv < 0

        ero = cv.erode(im, None)
        dil = cv.dilate(im, None)
        im1 = ero
        im1[m] = dil[m]
        im = np.uint8(im*(1.0 - blend) + im1*blend)

    return im


def preproc_image(im, ksize=3, n_iter=1, blend=0.5):
    """

    Args:
        im: image
        ksize: odd int
            kernel size for Gaussian blur
        n_iter: int
            count of iterations
        blend: float [0, ... ,1]
            blend coefficient
    Preprocessing function of fingerprint image.
    Preprocessing stages:
        1) Gaussian blur
        2) Histogramm equalization
        3) Coherence filter
        4) Otsu threshold

    Returns: ndarray
    Function returns
    """
    
    gaus_im = cv.GaussianBlur(im, (ksize,)*2, -1)
    eqhist_im = cv.equalizeHist(gaus_im)
    coh_im = coherence_filter(eqhist_im, n_iter=n_iter, blend=blend)
    thresh = cv.threshold(coh_im, 0, 255, cv.THRESH_OTSU)[1]
    
    return thresh