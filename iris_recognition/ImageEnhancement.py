import cv2


def run(img):
    img2 = img.copy()
    size = 32
    for i in range(2):
        for j in range(16):
            start_height = i*size
            end_height = start_height+size
            start_wid = j*size
            end_wid = start_wid+size
            grid = img2[start_height:end_height, start_wid:end_wid]
            img2[start_height:end_height, start_wid:end_wid] = cv2.equalizeHist(grid)

    return img2
