import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data/iris/001/1/001_1_2.bmp',0)
img = cv2.imread('data/iris/002/1/002_1_2.bmp',0)
plt.imshow(img, cmap = 'gray')
plt.show()
print(img)

y = np.argmin(np.sum(img, axis=0))
x = np.argmin(np.sum(img, axis=1))
pupil_pix = img[x,y]
pupil_pix
window = 110
img2 = img[x-window:x+window, y-window:y+window]
(thresh, im_bw) = cv2.threshold(img, pupil_pix, 255, cv2.THRESH_BINARY)
plt.imshow(im_bw, cmap = 'gray')
plt.show()
r1 = sum(img[x,:] == pupil_pix )
r2 = sum(img[:,y] == pupil_pix )
r = int((float(r1/2)+float(r2/2))/2)
cv2.circle(img, (y,x), r, 255)
plt.imshow(img, cmap = 'gray')
plt.show()


edges = cv2.Canny(im_bw,100,200)

plt.subplot(121),plt.imshow(im_bw,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
