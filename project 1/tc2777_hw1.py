import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.stats import gaussian_kde


lenna = Image.open('data/face_d2.jpg')
hist = np.array(lenna.histogram())
grey = lenna.convert("L")
grey_hist = np.array(grey.histogram())
fig = plt.figure()

xs = np.linspace(0, 255, 256)


hist = []
for i in range(256):
    hist.extend([int(xs[i])] * grey_hist[i])

hist = np.array(hist)
density = gaussian_kde(hist)
density.set_bandwidth(0.05)
density._compute_covariance()
plt.plot(xs,density(xs))
plt.hist(hist, bins=256, density=True)
a = np.ediff1d(density(xs))
asign = np.sign(a) + 1

signchange = (np.roll(asign, 1) - asign) == 2
signchange[0] = False
signchange = np.append(signchange, False)
K = sum(signchange)
initial = xs[signchange]

plt.show()

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
                calculator[closest] = np.append(calculator[closest],each_pixel)
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
print(K)
col = {0: (244, 66, 66), 1:(66, 244, 66), 2:(66, 66, 244), 3:(244, 66, 200), 4:(244, 244, 66), 5:(66, 244, 244)}
plt.imshow(lenna)
plt.show()
print(updated)
for j in range(K):
    x = x_indices[j]
    y = y_indices[j]
    length = len(x_indices[j])
    for k in range(length):
        lenna.putpixel((x[k], y[k]), col[j])

plt.imshow(lenna)

plt.show()
