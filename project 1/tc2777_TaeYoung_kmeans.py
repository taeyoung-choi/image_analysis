import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.stats import gaussian_kde


def get_peaks(grey_hist):
    """
    Calculates the number and location of peaks in a given histogram

    Parameters
    ----------
    grey_hist : numpy array of length 256

    each entry contains how many pixels are there for each gray value from 0 to 255

    Returns
    -------
    K : int
        the number of peaks in a given histogram
    initial : List(Int) of length K
        K gray values that peaks appear
    """

    xs = np.linspace(0, 255, 256)

    # transform the input that is appropriate for histogram
    hist = []
    for i in range(256):
        hist.extend([int(xs[i])] * grey_hist[i])
    hist = np.array(hist)

    # overlay a density curve to detect peaks
    density = gaussian_kde(hist)
    density.set_bandwidth(0.05)
    density._compute_covariance()

    # peaks appear when values stop increasing
    a = np.ediff1d(density(xs))

    # only count peaks not troughs
    asign = np.sign(a) + 1
    signchange = (np.roll(asign, 1) - asign) == 2
    # set the initial point as non-pick
    # this could occur if the last value is greater than the last value
    # np.roll moves kth element to k+nth position and
    # k-nth element to first position
    signchange[0] = False
    signchange = np.append(signchange, False)
    K = sum(signchange)
    initial = xs[signchange]

    # Saving histogram
    plt.plot(xs, density(xs))
    plt.hist(hist, bins=256, density=True)
    plt.savefig("data/hist.png")

    return K, initial


def k_means(K, initial, grey):
    """
    Runs K-means algorithm based on pixel values

    Parameters
    ----------
    K: number of clusters
    initial: K gray values that peaks appear
    grey: greyscale image

    Returns
    -------
    updated : List(Double)
        centroids calculated after algorithm terminates
    x_indices : dict
        x-coordinates of pixels for each cluster
    y_indices : dict
        y-coordinates of pixels for each cluster
    """

    # terminates if the updated centroids are the same as the previous iteration
    while True:
        # calculator stores pixel values of each cluster
        calculator = {}
        # x_indices stores x-coordinates of pixels for each cluster
        x_indices = {}
        # y_indices stores y-coordinates of pixels for each cluster
        y_indices = {}
        # iterating through all pixels
        (width, height) = grey.size
        for w in range(width):
            for h in range(height):
                each_pixel = grey.getpixel((w, h))
                # for each pixel calculate the closest centroid
                closest = np.argmin(abs(initial - each_pixel))
                # for new cluster create key in dictionary
                # otherwise append value to existing list of values
                if closest in calculator:
                    calculator[closest] = np.append(calculator[closest], each_pixel)
                    x_indices[closest] = np.append(x_indices[closest], w)
                    y_indices[closest] = np.append(y_indices[closest], h)
                else:
                    calculator[closest] = np.array([each_pixel])
                    x_indices[closest] = np.array([w])
                    y_indices[closest] = np.array([h])
        updated = []
        for i in range(K):
            # update centroid to the average value of clusters
            avg = np.mean(calculator[i])
            updated.insert(i, avg)
        # update centroids
        updated = np.array(updated)

        # algorithm terminates if centroids do not change
        if np.sum(initial - updated) == 0:
            break
        else:
            initial = updated
    return updated, x_indices, y_indices


def to_k_colors(image, K, x_indices, y_indices):
    """
    Change original image into K coloring

    Parameters
    ----------
    image : PIL Image
        original image
    K : int
        number of clusters
    x_indices : dict
        x-coordinates of pixels for each cluster
    y_indices : dict
        y-coordinates of pixels for each cluster
    """
    # candidate colors
    col = {0: (244, 66, 66), 1: (66, 244, 66), 2: (66, 66, 244), 3: (244, 66, 200),
           4: (244, 244, 66), 5: (66, 244, 244), 6: (244, 244, 244), 7: (66, 66, 66),
           8: (0, 0, 0), 9: (244, 244, 244)}

    # put RGB values for each cluster
    for j in range(K):
        x = x_indices[j]
        y = y_indices[j]
        length = len(x_indices[j])
        for k in range(length):
            image.putpixel((x[k], y[k]), col[j])
    # save image
    plt.imshow(image)
    plt.savefig("data/k_color.png")


def get_face(image, x_indices, y_indices):
    """
    Segment face from image

    Parameters
    ----------
    image : PIL Image
        original image
    x_indices : dict
        x-coordinates of pixels for each cluster
    y_indices : dict
        y-coordinates of pixels for each cluster
    """

    # the face should be colored blue
    # number of blue pixels
    blue = len(x_indices[2])

    # get a region in the image that scores the highest point
    best_score = 0
    # size of the best region, x-coordinate, y-coordinate
    best_conf = (0, 0, 0)

    # iterate through different regions of different sizes
    # j indicates the size of candidate region
    for j in range(10, min(image.size[0], image.size[1]), 100):
        # do not beyond the image size
        up_to_x = image.size[0] - j
        up_to_y = image.size[1] - j
        # iterate though different starting x, y coordinates
        for k in range(0, up_to_x, 5):
            for r in range(0, up_to_y, 10):
                score = 0
                # count the number of blue points within candidate region
                for q in range(len(y_indices[2])):
                    if (k < x_indices[2][q] < k + j) and (r < y_indices[2][q] < r + j):
                        score += 1
                # multiply two scores
                # first: (number of points within candidate region) / (total number of blue points)
                # second: (number of points within candidate region) / (number of pixels in candidate region)
                current = (score / blue) * (score / (j * j))

                # keep the best score
                if current > best_score:
                    best_score = current
                    best_conf = (j, k, r)
    # size of the best region, x-coordinate, y-coordinate
    (a, b, c) = best_conf
    # draw a square around the face
    for q in range(a):
        image.putpixel((b + q, c), (0, 0, 0))
        image.putpixel((b, c + q), (0, 0, 0))
        image.putpixel((b + a, c + q), (0, 0, 0))
        image.putpixel((b + q, c + a), (0, 0, 0))

    plt.imshow(image)
    plt.savefig("data/face_detected.png")


def main():
    image = Image.open('data/face_d2.jpg')
    image2 = Image.open('data/face_d2.jpg')
    # work with grayscale image
    grey = image.convert("L")
    grey_hist = np.array(grey.histogram())
    # calculate the number of clusters
    K, initial = get_peaks(grey_hist)
    # run K-means algorithm
    centroids, x_indices, y_indices = k_means(K, initial, grey)
    # turn image into K coloring image
    to_k_colors(image2, K, x_indices, y_indices)
    # draw a square around the face
    get_face(image, x_indices, y_indices)


if __name__ == "__main__":
    main()
