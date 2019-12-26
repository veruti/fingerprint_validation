import cv2 as cv
import numpy as np


def get_second_derivative(im, ksize=3):
    
    gxx = cv.Sobel(im, cv.CV_64F, 2, 0, ksize=ksize)
    gxy = cv.Sobel(im, cv.CV_64F, 1, 1, ksize=ksize)
    gyy = cv.Sobel(im, cv.CV_64F, 0, 2, ksize=ksize)
    
    return gxx, gxy, gyy


def coherence_filter(im, sigma = 15, str_sigma = 15, blend = 0.5, n_iter = 5):
    
    h, w = im.shape[:2]

    for i in range(n_iter):
        eigen = cv.cornerEigenValsAndVecs(im, str_sigma, 3)
        eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
        
        x, y = eigen[:,:,1,0], eigen[:,:,1,1]

        gxx, gxy, gyy = get_second_derivative(im, sigma)
        gvv = x*x*gxx + 2*x*y*gxy + y*y*gyy
        
        m = gvv < 0

        ero = cv.erode(im, None)
        dil = cv.dilate(im, None)
        im1 = ero
        im1[m] = dil[m]
        im = np.uint8(im*(1.0 - blend) + im1*blend)

    return im
