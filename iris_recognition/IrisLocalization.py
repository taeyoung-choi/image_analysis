import cv2
import numpy as np


def pupil_detection(img):
    """
    It calculates the center of a pupil by getting pixel coordinates and computing the mean.
    The radius is the average of the vertical length and the horizontal length from the center.
    Finally, it returns the center and the radius of a pupil.

    :param img: gray-scale eye image
    :return: pupil center coordinate and radius
    """

    (thresh, im_bw) = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15,15), np.uint8)
    # MORPH_CLOSE is useful in closing small holes inside the foreground objects, or small black points on the object.
    # (OpenCV 3.0.0-dev documentation)
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
    """
    Apply Canny Edge Detector, Remove noise, Apply Hough transformation to fit a circle to the outer boundary of a pupil

    :param img: gray-scale eye image
    :param pupil: pupil center, radius
    :param a: the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
    :param b: the accumulator threshold for the circle centers at the detection stage
    :return: outer boundary center and radius
    """
    col, row, r = pupil

    # remove noise by blurring the image
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    x = int(round(len(blurred[0]) * 0.3))
    y = int(round(len(blurred) * 0.3))

    # Resize the image by 30% in order to speed up the computation.
    resized_image = cv2.resize(blurred, (x, y))
    # Apply Canny edge detector.
    edges = cv2.Canny(resized_image, a, b)

    # rescale the center and the radius accordingly
    r = int(round(r*0.3))
    col, row = int(round(col*0.3)), int(round(row*0.3))

    n, m = edges.shape
    # Remove the pupil area to remove noise created from eyelashes and the pupil edge
    # Removing a circular area centered at the pupil center with the radius of 12 pixels greater than the pupil
    # radius works well for our purposes
    r1 = r + 12
    y1, x1 = np.ogrid[-row:n-row, -col:m-col]
    mask = x1*x1 + y1*y1 <= r1*r1
    edges[mask] = 0

    # trying to locate first edge found below the eye
    lower_removal = edges[row+r+8:, col]
    l_bound = np.argmax(lower_removal == 255)+row+r+8
    # trying to locate first edge found above the eye
    upper_removal = edges[row-r-8:, col]
    u_bound = row-r-8+np.argmin(upper_removal == 255)
    # remove upper and lower parts
    edges[:u_bound, ] = 0
    edges[l_bound:, ] = 0

    # trying to locate first edge found left of the eye
    left = edges[row, :col][::-1]
    left = np.argmax(left == 255)

    # trying to locate first edge found left of the eye
    right = edges[row, col:]
    right = np.argmax(right == 255)


    # we might have no edges found along the center
    if (col - left - 2 > 15) and left != 0:
        edges[:, :(col - left - 2)] = 0
    # if so, remove the area where no eyes have been found in the training data
    else:
        edges[:, :20] = 0

    # we might have no edges found along the center
    if (x - (col + right + 2)) > 15 and right != 0:
        edges[:, (col + right + 2):] = 0
    # if so, remove the area where no eyes have been found in the training data
    else:
        edges[:, 90:] = 0

    # Apply Hough transformation to fit a circle for the outer boundary of a pupil
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1,70,param1=20,param2=3,minRadius=30,maxRadius=35)
    if circles is not None:
        # scale back
        circles = np.uint16(np.around(circles/0.3))
        for i in circles[0, :]:
            return i[0], i[1], i[2]
    else:
        # manually fit a circle centered at the pupil circle with the average radius of outer bounds of eyes
        # scale back
        col = int(round(col/0.3))
        row = int(round(row/0.3))
        r = int(round(r/0.3)) + 60
        return col, row, r


def run(img):
    pupil = pupil_detection(img)
    outer = outer_boundary(img, pupil, 50, 100)

    return pupil, outer
