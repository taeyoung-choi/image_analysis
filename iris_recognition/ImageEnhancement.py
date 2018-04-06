import cv2
import matplotlib.pyplot as plt

def run(img):
    """
    It divides the normalized image into 32 by 32 grids and equalizes the histogram of each grid.

    :param img: normalized iris image
    :return: enhance image by histogram equalization
    """
    # do not change the original image
    img2 = img.copy()
    size = 32
    for i in range(2):
        for j in range(16):
            # Define each 32 by 32 grid iteratively
            start_height = i*size
            end_height = start_height+size
            start_wid = j*size
            end_wid = start_wid+size
            grid = img2[start_height:end_height, start_wid:end_wid]
            # Histogram Equalization
            img2[start_height:end_height, start_wid:end_wid] = cv2.equalizeHist(grid)
    plt.imshow(img2, cmap="gray")
    plt.savefig("plot/enhanced.png")

    return img2
