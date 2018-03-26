import cv2
import numpy as np
from matplotlib import pyplot as plt

def pupil_detection(img):
    plt.imshow(img, cmap='gray')
    plt.show()
    col = np.argmin(np.sum(img, axis=0))
    row = np.argmin(np.sum(img, axis=1))
    pupil_pix = img[row,col]
    (thresh, im_bw) = cv2.threshold(img, pupil_pix, 255, cv2.THRESH_BINARY)

    length = np.array(range(len(im_bw)))
    width = np.array(range(len(im_bw[0])))
    col = int(round(np.mean(width[np.mean(im_bw, axis=0) != 255])))
    row = int(round(np.mean(length[np.mean(im_bw, axis=1) != 255])))

    r1 = sum(im_bw[row,:] == 0)
    r2 = sum(im_bw[:,col] == 0)
    r = int(round((r1+r2)/4))
    return (col, row, r)

def outer_boundary(img, col, row, r, a, b):
    img2 = img.copy()
    cv2.circle(img2,(col,row),r,255,2)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    x = int(len(blurred[0])*0.3)
    y = int(len(blurred)*0.3)
    resized_image = cv2.resize(blurred, (x, y))
    edges = cv2.Canny(resized_image, a, b)

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

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1,70,param1=20,param2=3,minRadius=30,maxRadius=35)
    circles = np.uint16(np.around(circles/0.3))

    for i in circles[0,:]:
        cv2.circle(img2,(i[0],i[1]),i[2],255,2)
        plt.imshow(img2, cmap='gray')
        plt.show()
        return (i[0],i[1],i[2])


def iris_normalization(img, in_col, in_row, in_r, out_col, out_row, out_r):
    r = int(round(out_r - np.sqrt((in_col-out_col)**2 + (in_row-out_row)**2)))
    max_r = r - in_r
    M = 64
    N = 512
    normalized = []
    for y in range(M):
        row_pix = []
        for x in range(N):
            theta = float(2*np.pi*x/N)
            hypotenuse = float(max_r*y/M) + in_r
            col = int(round((np.cos(theta) * hypotenuse) + in_col))
            row = int(round((np.sin(theta) * hypotenuse) + in_row))
            row_pix.append(img[row,col])
        normalized.append(row_pix)

    return np.array(normalized)

def illumination_adjust(normalized_img):
    height, width = normalized_img.shape
    illumination = []
    for i in range(4):
        each_row = []
        for j in range(32):
            start_height = i*16
            end_height = start_height+16
            start_wid = j*16
            end_wid = start_wid+16
            avg = np.mean(normalized_img[start_height:end_height,start_wid:end_wid])
            each_row.append(round(avg))
        illumination.append(each_row)
    illumination = np.array(illumination)
    return cv2.resize(np.array(illumination), (width, height), cv2.INTER_CUBIC)

def enhencement(img):
    img2 = img.copy()
    for i in range(2):
        for j in range(16):
            start_height = i*32
            end_height = start_height+32
            start_wid = j*32
            end_wid = start_wid+32
            grid = img2[start_height:end_height,start_wid:end_wid]
            img2[start_height:end_height,start_wid:end_wid] = cv2.equalizeHist(grid)
    return img2

def spatial_filter(x,y,delta_x,delta_y,f):
    modul_fun = np.cos(2*np.pi*f*np.hypot(x,y))
    gaus_envelop = (x**2)/(delta_x**2)+(y**2)/(delta_y**2)
    filter = 1/(2*np.pi*delta_x*delta_y)*np.exp(-0.5*(gaus_envelop))*modul_fun

    return filter

def to_feature_vec(img, block_size=8):
    roi = img[:48,:]
    len, wid = roi.size/block_size
    vec_size = len*wid*4
    feature_vec = np.empty((0,vec_size))
    for i in range(len):
        for j in range(wid):
            start_height = i*block_size
            end_height = start_height+block_size
            start_wid = j*block_size
            end_wid = start_wid+block_size
            block1=[]
            block2=[]
            for m in range(start_height, end_height):
                for n in range(start_wid, end_wid):
                    block1.append(roi[m,n]*spatial_filter)
    

img = cv2.imread('data/iris/001/1/001_1_1.bmp',0)
img = cv2.imread('data/iris/012/1/012_1_1.bmp',0)
in_col, in_row, in_r = pupil_detection(img)
out_col, out_row, out_r = outer_boundary(img, in_col, in_row, in_r, 50, 100)
normalized = iris_normalization(img, in_col, in_row, in_r, out_col, out_row, out_r)
plt.imshow(normalized, cmap='gray')
plt.show()

illumination = illumination_adjust(normalized)
illum_adjusted = np.array(normalized-np.round(illumination))
plt.imshow(illum_adjusted, cmap='gray')
plt.show()

illum_adjusted = (illum_adjusted - np.min(illum_adjusted)).astype(np.uint8)
plt.imshow(illum_adjusted, cmap='gray')
plt.show()

enhenced = enhencement(illum_adjusted)
plt.imshow(enhenced, cmap='gray')
plt.show()
