import cv2


def run(img):
    """
    It divides the normalized image into 16 by 16 grids and equalizes the histogram of each grid.

    :param img: normalized iris image
    :return: enhance image by histogram equalization
    """
    # do not change the original image
    img2 = img.copy()
    size = 16
    for i in range(4):
        for j in range(32):
            # Define each 16 by 16 grid iteratively
            start_height = i*size
            end_height = start_height+size
            start_wid = j*size
            end_wid = start_wid+size
            grid = img2[start_height:end_height, start_wid:end_wid]
            # Histogram Equalization
            img2[start_height:end_height, start_wid:end_wid] = cv2.equalizeHist(grid)

    return img2
