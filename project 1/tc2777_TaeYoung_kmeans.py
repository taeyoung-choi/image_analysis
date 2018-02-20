import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.stats import gaussian_kde


def to_grey_hist(image):
    grey = image.convert("L")
    grey_hist = np.array(grey.histogram())

    return grey_hist


def get_peaks(grey_hist):
    xs = np.linspace(0, 255, 256)
    hist = []
    for i in range(256):
        hist.extend([int(xs[i])] * grey_hist[i])

    hist = np.array(hist)
    density = gaussian_kde(hist)
    density.set_bandwidth(0.05)
    density._compute_covariance()
    plt.plot(xs, density(xs))
    plt.hist(hist, bins=256, density=True)
    plt.savefig("data/hist.png")
    a = np.ediff1d(density(xs))
    asign = np.sign(a) + 1

    signchange = (np.roll(asign, 1) - asign) == 2
    signchange[0] = False
    signchange = np.append(signchange, False)
    K = sum(signchange)
    initial = xs[signchange]

    return K, initial


def k_means(K, initial, grey):
    while True:
        calculator = {}
        x_indices = {}
        y_indices = {}
        (width, height) = grey.size
        for w in range(width):
            for h in range(height):
                each_pixel = grey.getpixel((w, h))
                closest = np.argmin(abs(initial - each_pixel))
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
            avg = np.mean(calculator[i])
            updated.insert(i, avg)
        updated = np.array(updated)

        if np.sum(initial - updated) == 0:
            break
        else:
            initial = updated
    return updated, x_indices, y_indices


def to_k_colors(image, K, x_indices, y_indices):
    col = {0: (244, 66, 66), 1: (66, 244, 66), 2: (66, 66, 244), 3: (244, 66, 200),
           4: (244, 244, 66), 5: (66, 244, 244), 6: (244, 244, 244), 7: (66, 66, 66),
           8: (0, 0, 0), 9: (244, 244, 244)}

    for j in range(K):
        x = x_indices[j]
        y = y_indices[j]
        length = len(x_indices[j])
        for k in range(length):
            image.putpixel((x[k], y[k]), col[j])

    plt.imshow(image)
    plt.savefig("data/k_color.png")


def get_face(image, x_indices, y_indices):
    blue = len(x_indices[2])
    best_score = 0
    best_conf = (0, 0, 0)
    for j in range(10, min(image.size[0], image.size[1]), 100):
        up_to_x = image.size[0] - j
        up_to_y = image.size[1] - j
        for k in range(0, up_to_x, 5):
            for r in range(0, up_to_y, 10):
                score = 0
                for q in range(len(y_indices[2])):
                    if (k < x_indices[2][q] < k + j) and (r < y_indices[2][q] < r + j):
                        score += 1
                current = (score / blue) * (score / (j * j))
                if current > best_score:
                    best_score = current
                    best_conf = (j, k, r)
    (a, b, c) = best_conf
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
    grey = image.convert("L")
    grey_hist = np.array(grey.histogram())
    K, initial = get_peaks(grey_hist)
    centroids, x_indices, y_indices = k_means(K, initial, grey)
    to_k_colors(image2, K, x_indices, y_indices)
    get_face(image, x_indices, y_indices)


if __name__ == "__main__":
    main()
