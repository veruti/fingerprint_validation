

def find_contours_result(res):
    """

    Args:
        res: results of findContours function
    Function return contours and hierarchy

    Returns: contours, hierarchy

    """
    if len(res) == 2:
        return res[0], res[1]
    elif len(res) == 3:
        return res[1], res[2]
    else:
        return None