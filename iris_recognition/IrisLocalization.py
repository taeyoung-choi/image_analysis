import cv2
import numpy as np

def pupil_detection(img):
    (thresh, im_bw) = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)

    kernel = np.ones((15,15), np.uint8)
    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)

    length = np.array(range(len(im_bw)))
    width = np.array(range(len(im_bw[0])))
    col = int(round(np.mean(width[np.mean(im_bw, axis=0) != 255])))
    row = int(round(np.mean(length[np.mean(im_bw, axis=1) != 255])))

    r1 = sum(im_bw[row, :] == 0)
    r2 = sum(im_bw[:, col] == 0)
    r = int(round((r1+r2)/4))
    return col, row, r


def outer_boundary(img, pupil, a, b):
    col, row, r = pupil
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    x = int(round(len(blurred[0])*0.3))
    y = int(round(len(blurred)*0.3))
    resized_image = cv2.resize(blurred, (x, y))
    edges = cv2.Canny(resized_image, a, b)

    r = int(round(r*0.3))
    col, row = int(round(col*0.3)), int(round(row*0.3))

    n, m = edges.shape
    r1 = r + 12
    y1, x1 = np.ogrid[-row:n-row, -col:m-col]
    mask = x1*x1 + y1*y1 <= r1*r1
    edges[mask] = 0


    lower_removal = edges[row+r+8:, col]
    l_bound = np.argmax(lower_removal == 255)+row+r+8
    upper_removal = edges[row-r-8:, col]
    u_bound = row-r-8+np.argmin(upper_removal == 255)
    edges[:u_bound, ] = 0
    edges[l_bound:, ] = 0

    left = edges[row, :col][::-1]
    right = edges[row, col:]
    left = np.argmax(left == 255)
    right = np.argmax(right == 255)

    if left != 0:
        if col-left-2 > 15:
            edges[:,:(col-left-2)] = 0
        else:
            edges[:,:20] = 0

    else:
        edges[:, :20] = 0
    if (x-(col+right+2)) > 15 and right != 0:
        edges[:, (col+right+2):] = 0
    else:
        edges[:, 90:] = 0

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1,70,param1=20,param2=3,minRadius=30,maxRadius=35)
    if circles is not None:
        circles = np.uint16(np.around(circles/0.3))

        for i in circles[0,:]:

            return i[0], i[1], i[2]
    else:
        col = int(round(col/0.3))
        row = int(round(row/0.3))
        r = int(round(r/0.3)) + 60

        return col, row, r


def run(img):
    pupil = pupil_detection(img)
    outer = outer_boundary(img, pupil, 50, 100)

    return pupil, outer
