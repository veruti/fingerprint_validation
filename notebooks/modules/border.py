def left_right_border(img, left_perc=0.15, right_perc=0.15):
    
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
    
    cs = np.cumsum(np.sum(img, axis=1))
    cs = cs / cs[-1]
    i1 = np.argmax(cs > top_perc)
    i2 = cs.size - np.argmax(cs[::-1] < 1-bottom_perc)
    
    contour_top = img.copy()
    contour_bottom = img.copy()
    
    contour_top[i1:, :] = 0
    contour_bottom[:i2, :] = 0   
    
    return contour_top, contour_bottom