import cv2 as cv


def invers_color(image):
    if image.mean() > 127.5:
        inv_image = cv.bitwise_not(image)
    else:
        inv_image = image.copy()

    return inv_image


def crop_fp_image(fp_image, bound_const=5, with_golden_ratio=True):

    # invert colors (black -> white, white -> black)
    # and binarize image

    temp_image = fp_image

    # get contours and unite contours
    _1, contours, _2 = cv.findContours(image=temp_image, mode=cv.RETR_EXTERNAL,
                                       method=cv.CHAIN_APPROX_SIMPLE)
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
    if with_golden_ratio:
        resized_image = get_golden_ratio_image(resized_image)
        
    resized_image = cv.copyMakeBorder(resized_image, bound_const, bound_const, 
                      bound_const, bound_const, cv.BORDER_CONSTANT, value=0)
    
    return resized_image


def get_propotion(im):
    propotion = im.shape[0] / im.shape[1]
    return propotion


def get_golden_ratio_image(im):
    phi = (np.sqrt(5) + 1) / 2
    pr = get_propotion(im)

    if pr > phi:
        new_height = int(im.shape[1] * phi)
        new_im = im[:new_height, :]

        return new_im
    else:
        return im
    
    
def resize_fp_image(im, new_width=400, interpolation=cv.INTER_LINEAR):
    
    height, width = im.shape
    k = height/width

    resized_image = cv.resize(im, (new_width, int(new_width*k)), interpolation=cv.INTER_LINEAR)
    
    return resized_image


def standardize_image(im, bound_const=5, new_width=400, interpolation=cv.INTER_LINEAR, with_golden_ratio=True):
    
    inv_im = invers_color(im)
    crop_im = crop_fp_image(inv_im, bound_const=bound_const,
                            with_golden_ratio=with_golden_ratio)
    resized_im = resize_fp_image(crop_im, new_width=new_width,
                                 interpolation=interpolation)
    
    return resized_im