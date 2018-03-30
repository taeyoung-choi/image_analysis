import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la

def pupil_detection(img):
    # plt.imshow(img, cmap='gray')
    # plt.show()
    col = np.argmin(np.sum(img, axis=0))
    row = np.argmin(np.sum(img, axis=1))
    pupil_pix = img[row,col]
    # print(pupil_pix)
    (thresh, im_bw) = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)
    # plt.imshow(im_bw, cmap='gray')
    # plt.show()
    kernel = np.ones((15,15),np.uint8)
    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(im_bw, cmap='gray')
    # plt.show()

    length = np.array(range(len(im_bw)))
    width = np.array(range(len(im_bw[0])))
    col = int(round(np.mean(width[np.mean(im_bw, axis=0) != 255])))
    row = int(round(np.mean(length[np.mean(im_bw, axis=1) != 255])))

    r1 = sum(im_bw[row,:] == 0)
    r2 = sum(im_bw[:,col] == 0)
    r = int(round((r1+r2)/4))
    return (col, row, r)

def outer_boundary(img, col, row, r, a, b):
    # print(col,row)
    # img2 = img.copy()
    # cv2.circle(img2,(col,row),r,255,2)

    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    x = int(round(len(blurred[0])*0.3))
    y = int(round(len(blurred)*0.3))
    resized_image = cv2.resize(blurred, (x, y))
    edges = cv2.Canny(resized_image, a, b)
    # plt.imshow(edges, cmap='gray')
    # plt.show()
    r = int(round(r*0.3))
    col, row = int(round(col*0.3)), int(round(row*0.3))

    n, m = edges.shape
    r1 = r + 12
    y1,x1 = np.ogrid[-row:n-row, -col:m-col]
    mask = x1*x1 + y1*y1 <= r1*r1
    edges[mask] = 0

    # plt.imshow(edges, cmap='gray')
    # plt.show()
    lower_removal = edges[row+r+8:, col]
    l_bound = np.argmax(lower_removal==255)+row+r+8
    upper_removal = edges[row-r-8:, col]
    u_bound = row-r-8+np.argmin(upper_removal==255)
    edges[:u_bound,] = 0
    edges[l_bound:,] = 0
    # plt.imshow(edges, cmap='gray')
    # plt.show()
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
        edges[:,:20] = 0
    if (x-(col+right+2)) > 15 and right != 0:
        edges[:,(col+right+2):] = 0
    else:
        edges[:,90:] = 0

    # plt.imshow(edges, cmap='gray')
    # plt.show()
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1,70,param1=20,param2=3,minRadius=30,maxRadius=35)
    if circles is not None:
        circles = np.uint16(np.around(circles/0.3))

        for i in circles[0,:]:
            # cv2.circle(img2,(i[0],i[1]),i[2],255,2)
            # plt.imshow(img2, cmap='gray')
            # plt.show()
            return (i[0],i[1],i[2])
    else:
        col = int(round(col/0.3))
        row = int(round(row/0.3))
        r = int(round(r/0.3)) + 60
        # cv2.circle(img2,(col,row),r,255,2)
        # plt.imshow(img2, cmap='gray')
        # plt.show()
        return (col,row,r)


def iris_normalization(img, in_col, in_row, in_r, out_col, out_row, out_r):
    r = int(round(out_r - np.hypot(in_col-out_col, in_row-out_row)))
    maxh, maxw = img.shape
    r2 = min(in_col, in_row, maxh-in_row, maxw-in_col)
    r = min(r, r2)
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

def to_feature_vec(image, block_size=7):
    roi = np.array(image[:49,:511])
    (len, wid) = roi.shape
    vec_size = int((len*wid*4)/(block_size**2))
    channel1 = np.empty((49,wid))
    channel2 = np.empty((49,wid))
    each_side = int((block_size-1)/2)
    feature_vec = []
    for i in range(len):
        for j in range(wid):
            each_pix1 = 0
            each_pix2 = 0
            for m in range(i-each_side, i+each_side+1):
                for n in range(j-each_side, j+each_side+1):
                    if (m > 0 and m < len) and (n > 0 and n < wid):
                        each_pix1 += roi[m, n]*spatial_filter(i-m, j-n, 3, 1.5, 1/1.5)
                        each_pix2 += roi[m, n]*spatial_filter(i-m, j-n, 4.5, 1.5, 1/1.5)
            channel1[i,j] = each_pix1
            channel2[i,j] = each_pix2

    for i in range(int(len/block_size)):
        for j in range(int(wid/block_size)):
            start_height = i*block_size
            end_height = start_height+block_size
            start_wid = j*block_size
            end_wid = start_wid+block_size
            grid1 = channel1[start_height:end_height, start_wid:end_wid]
            grid2 = channel2[start_height:end_height, start_wid:end_wid]

            absolute = np.absolute(grid1)
            mean = np.mean(absolute)
            feature_vec.append(mean)
            std = np.mean(np.absolute(absolute-mean))
            feature_vec.append(std)

            absolute = np.absolute(grid2)
            mean = np.mean(absolute)
            feature_vec.append(mean)
            std = np.mean(np.absolute(absolute-mean))
            feature_vec.append(std)

    return feature_vec


def get_train_feature():
    x_train = []
    y_train = []
    for i in range(1,109):
        print(i)
        for j in range(1,4):
            eye_num = str(i).zfill(3)
            path = 'data/iris/{}/1/{}_1_{}.bmp'.format(eye_num,eye_num,j)
            img = cv2.imread(path,0)
            in_col, in_row, in_r = pupil_detection(img)
            out_col, out_row, out_r = outer_boundary(img, in_col, in_row, in_r, 50, 100)
            normalized = iris_normalization(img, in_col, in_row, in_r, out_col, out_row, out_r)
    #         illumination = illumination_adjust(normalized)
    #         illum_adjusted = np.array(normalized-np.round(illumination))
    #         illum_adjusted = (illum_adjusted - np.min(illum_adjusted)).astype(np.uint8)
            enhenced = enhencement(normalized)
            # plt.imshow(enhenced, cmap='gray')
            # plt.show()
            feature = to_feature_vec(enhenced)
            x_train.append(feature)
            y_train.append(i)
    #
    np.savetxt('train_feature.csv', x_train, delimiter=',')
    np.savetxt('train_label.csv', y_train, delimiter=',')

    return x_train, y_train

def get_test_feature():
    x_test = []
    y_test = []
    for i in range(1,109):
        print(i)
        for j in range(1,5):
            eye_num = str(i).zfill(3)
            path = 'data/iris/{}/2/{}_2_{}.bmp'.format(eye_num,eye_num,j)
            img = cv2.imread(path,0)
            in_col, in_row, in_r = pupil_detection(img)
            out_col, out_row, out_r = outer_boundary(img, in_col, in_row, in_r, 50, 100)
            normalized = iris_normalization(img, in_col, in_row, in_r, out_col, out_row, out_r)
            # illumination = illumination_adjust(normalized)
            # illum_adjusted = np.array(normalized-np.round(illumination))
            # illum_adjusted = (illum_adjusted - np.min(illum_adjusted)).astype(np.uint8)
            enhenced = enhencement(normalized)
            # plt.imshow(enhenced, cmap='gray')
            # plt.show()
            feature = to_feature_vec(enhenced)
            x_test.append(feature)
            y_test.append(i)
    #
    np.savetxt('test_feature.csv', x_test, delimiter=',')
    np.savetxt('test_label.csv', y_test, delimiter=',')
    return x_test, y_test

def projection_matrix(x_train):
    training_mean = np.mean(np.array(x_train), axis=0)
    training_mean.shape = (2044,1)

    between_class = np.zeros((2044,2044))
    within_class = np.zeros((2044,2044))
    for i in range(108):
        each_class = np.array(x_train)[3*i:3*i+3, ]
        class_mean = np.mean(each_class, axis=0)
        class_mean.shape = (2044,1)
        diff = class_mean - training_mean
        between_class += diff.dot(diff.transpose())
        for j in range(3):
            each_row = each_class[j,]
            each_row.shape = (2044,1)
            diff2 = each_row - class_mean
            within_class += diff2.dot(diff2.transpose())

    eigen_vals, eigen_vecs = np.linalg.eigh(np.linalg.inv(within_class).dot(between_class))
    idx = eigen_vals.argsort()[::-1]
    eigen_vecs = eigen_vecs[:,idx]
    return eigen_vecs


# def main():
    x_train, y_train =get_train_feature()
    x_test, y_test =get_test_feature()

x_train = np.genfromtxt('train_feature.csv', delimiter=',')
y_train = np.genfromtxt('train_label.csv', delimiter=',')
x_test = np.genfromtxt('test_feature.csv', delimiter=',')
y_test = np.genfromtxt('test_label.csv', delimiter=',')

x_train.shape

accuracy_1 = []
accuracy_2 = []
accuracy_3 = []
eigen_vecs = projection_matrix(x_train)
x_train = np.array(x_train)
x_test = np.array(x_test)
for k in range(5,400,5):
    proj_mat = eigen_vecs[:,:k]
    train_reduced = np.matmul(x_train,proj_mat)
    test_reduced = np.matmul(x_test,proj_mat)

    y_predicted_1 = []
    y_predicted_2 = []
    y_predicted_3 = []
    for j in range(432):
        opt1 = 99999
        opt2 = 99999
        opt3 = 99999
        for i in range(324):
            val1 = np.sum(np.abs(train_reduced[i,:] - test_reduced[j,:]))
            val2 = np.sum(np.power(train_reduced[i,:] - test_reduced[j,:], 2))
            train = train_reduced[i,:]
            test = test_reduced[j,:]
            train.shape = (k,1)
            test.shape = (1,k)
            val3 = 1 - test.dot(train)/(la.norm(train)*la.norm(test))
            if val1 < opt1:
                opt1 = val1
                inx1 = y_train[i]
            if val2 < opt2:
                opt2 = val2
                inx2 = y_train[i]
            if val3 < opt3:
                opt3 = val3
                inx3 = y_train[i]
        y_predicted_1.append(inx1)
        y_predicted_2.append(inx2)
        y_predicted_3.append(inx3)
    accuracy_1.append(np.mean(np.array(y_test) == np.array(y_predicted_1)))
    accuracy_2.append(np.mean(np.array(y_test) == np.array(y_predicted_2)))
    accuracy_3.append(np.mean(np.array(y_test) == np.array(y_predicted_3)))

x = []
for i in range(5,400,5):
    x.append(i)

np.savetxt('distance1.csv', accuracy_1, delimiter=',')
np.savetxt('distance2.csv', accuracy_2, delimiter=',')
np.savetxt('distance3.csv', accuracy_3, delimiter=',')

plt.plot(x,accuracy_1,x,accuracy_2,x,accuracy_3)
plt.show()
