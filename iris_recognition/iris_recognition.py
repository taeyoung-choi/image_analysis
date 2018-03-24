import cv2
import numpy as np
from matplotlib import pyplot as plt

def pupil_detection(img):
    col = np.argmin(np.sum(img, axis=0))
    row = np.argmin(np.sum(img, axis=1))
    pupil_pix = img[row,col]
    (thresh, im_bw) = cv2.threshold(img, pupil_pix, 255, cv2.THRESH_BINARY)
    col = np.argmin(np.sum(im_bw, axis=0))
    row = np.argmin(np.sum(im_bw, axis=1))
    r1 = sum(im_bw[row,:] == 0)
    r2 = sum(im_bw[:,col] == 0)
    r = int((r1+r2)/4)

    return col, row, r

def outer_boundary(img, col, row, r, a, b):
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    x = int(len(blurred[0])*0.3)
    y = int(len(blurred)*0.3)
    resized_image = cv2.resize(blurred, (x, y))
    edges = cv2.Canny(resized_image, a, b)
    plt.imshow(edges, cmap='gray')
    plt.show()
    r = int(r*0.3)
    col, row = int(col*0.3), int(row*0.3)
    edges[row-r-8:row+r+8, col-r-13:col+r+13] = 0

    lower_removal = edges[row+r+4:, col]
    l_bound = np.argmax(lower_removal==255)+row+r+4
    upper_removal = edges[row-r-4:, col]
    u_bound = row-r-4+np.argmin(upper_removal==255)
    edges[:u_bound,] = 0
    edges[l_bound:,] = 0

    left = edges[row, :col][::-1]
    right = edges[row, col:]

    left = np.argmax(left == 255)
    right = np.argmax(right == 255)
    if left != 0:
        edges[:,:(col-left-2)] = 0
    if right != 0:
        edges[:,(col+right+2):] = 0
    plt.imshow(edges, cmap='gray')
    plt.show()

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1,70,param1=20,param2=3,minRadius=30,maxRadius=35)
    circles = np.uint16(np.around(circles/0.3))
    for i in circles[0,:]:
        cv2.circle(img,(i[0],i[1]),i[2],0,2)

    plt.imshow(img, cmap='gray')
    plt.show()

img = cv2.imread('data/iris/001/1/001_1_1.bmp',0)
img = cv2.imread('data/iris/014/1/014_1_2.bmp',0)
col, row, r = pupil_detection(img)
outer_boundary(img, col, row, r, 50, 100)
