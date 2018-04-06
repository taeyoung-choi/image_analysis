import numpy as np


def spatial_filter(x, y, delta_x, delta_y, f):
    """
    Define new spatial filters to capture local details of the iris

    :param x: x-coordinate
    :param y: y-coordinate
    :param delta_x: space constants of the Gaussian envelope along the x axis
    :param delta_y: space constants of the Gaussian envelope along the y axis
    :param f: frequency of the sinusoidal function
    :return: filtered image
    """
    modul_fun = np.cos(2*np.pi*f*np.hypot(x, y))
    gaus_envelop = (x**2)/(delta_x**2)+(y**2)/(delta_y**2)
    spatia_filter = 1/(2*np.pi*delta_x*delta_y)*np.exp(-0.5*gaus_envelop)*modul_fun

    return spatia_filter


def to_feature_vec(image, block_size=7):
    """
    Converts image to a feature vector.
    A feature vector represents the mean and the average absolute deviation of each 7 by 7 filtered block.

    :param image: enhanced/normalized image
    :param block_size: filter block size
    :return: feature vector
    """
    # multiples of 7 which is the default block size
    roi = np.array(image[:42, :511])
    len, wid = roi.shape
    channel1 = np.empty((len, wid))
    channel2 = np.empty((len, wid))
    each_side = int((block_size-1)/2)
    feature_vec = []
    for i in range(len):
        for j in range(wid):
            each_pix1 = 0
            each_pix2 = 0
            for m in range(i-each_side, i+each_side+1):
                for n in range(j-each_side, j+each_side+1):
                    # Around the boundaries of the image, it applies the zero-padding.
                    if (0 < m < len) and (0 < n < wid):
                        each_pix1 += roi[m, n]*spatial_filter(i-m, j-n, 3, 1.5, 1/1.5)
                        each_pix2 += roi[m, n]*spatial_filter(i-m, j-n, 4.5, 1.5, 1/1.5)
            channel1[i, j] = each_pix1
            channel2[i, j] = each_pix2

    for i in range(int(len/block_size)):
        for j in range(int(wid/block_size)):
            # Define each 7 by 7 filtered block iteratively
            start_height = i*block_size
            end_height = start_height+block_size
            start_wid = j*block_size
            end_wid = start_wid+block_size
            grid1 = channel1[start_height:end_height, start_wid:end_wid]
            grid2 = channel2[start_height:end_height, start_wid:end_wid]

            # Channel 1
            absolute = np.absolute(grid1)
            # mean
            mean = np.mean(absolute)
            feature_vec.append(mean)
            # absolute deviation
            std = np.mean(np.absolute(absolute-mean))
            feature_vec.append(std)

            # Channel 2
            absolute = np.absolute(grid2)
            # mean
            mean = np.mean(absolute)
            feature_vec.append(mean)
            # absolute deviation
            std = np.mean(np.absolute(absolute-mean))
            feature_vec.append(std)

    return feature_vec


def run(img, d):
    return to_feature_vec(img, d)
