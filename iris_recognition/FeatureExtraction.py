import numpy as np


def spatial_filter(x, y, delta_x, delta_y, f):
    modul_fun = np.cos(2*np.pi*f*np.hypot(x, y))
    gaus_envelop = (x**2)/(delta_x**2)+(y**2)/(delta_y**2)
    filter = 1/(2*np.pi*delta_x*delta_y)*np.exp(-0.5*gaus_envelop)*modul_fun

    return filter


def to_feature_vec(image, block_size=7):
    roi = np.array(image[:49, :511])
    len, wid = roi.shape
    channel1 = np.empty((49, wid))
    channel2 = np.empty((49, wid))
    each_side = int((block_size-1)/2)
    feature_vec = []
    for i in range(len):
        for j in range(wid):
            each_pix1 = 0
            each_pix2 = 0
            for m in range(i-each_side, i+each_side+1):
                for n in range(j-each_side, j+each_side+1):
                    if (0 < m < len) and (0 < n < wid):
                        each_pix1 += roi[m, n]*spatial_filter(i-m, j-n, 3, 1.5, 1/1.5)
                        each_pix2 += roi[m, n]*spatial_filter(i-m, j-n, 4.5, 1.5, 1/1.5)
            channel1[i, j] = each_pix1
            channel2[i, j] = each_pix2

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


def run(img):
    return to_feature_vec(img)
