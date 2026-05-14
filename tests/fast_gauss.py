import numpy as np
from numba import njit
import cv2
import scipy.ndimage as ndi
from skimage._shared.filters import gaussian
from numba.typed import List
from time import time

@njit
def filt(gray, gausses):
    h, w = gray.shape
    octaves = []

    for gauss in gausses:
        new_image = np.zeros_like(gray)
        blurred = np.zeros_like(gray).T

        rad = len(gauss[0]) // 2

        # horizontal pass
        for i in range(h):
            for j in range(w):
                acc = 0.0
                norm = 0.0
                ker = gauss[i+j]
                for k in range(-rad, rad + 1):
                    jj = j + k
                    if jj < 0:
                        jj = -jj
                    elif jj >= w:
                        jj = 2*w - jj - 2
                    weight = ker[k + rad]
                    acc += gray[i, jj] * weight
                    norm += weight

                new_image[i, j] = acc #/ max(norm, 1e-9)

        new_image = new_image.T
        # vertical pass
        for i in range(w):
            for j in range(h):
                acc = 0.0
                norm = 0.0
                ker = gauss[i+j]

                for k in range(-rad, rad + 1):
                    jj = j + k
                    if jj < 0:
                        jj = -jj
                    elif jj >= h:
                        jj = 2*h - jj - 2
                    weight = ker[k + rad]
                    acc += new_image[i, jj] * weight
                    norm += weight

                blurred[i, j] = acc #/ max(norm, 1e-9)

        octaves.append(blurred.T)
    return octaves

#@njit
def make_filts(s):
    truncate = 4.0
    s = max(float(s * 1.6), 1e-12)
    rad = max(1, int(truncate * s + 0.5))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * s * s))
    k = k / np.sum(k)
    return k 

img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sigma = 1.6

t0 = time()

for i in range(1):
    img = cv2.imread("input.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    stock_gauss = ndi.gaussian_filter(gray, sigma)

t1 = time()

for i in range(1):
    img = cv2.imread("input.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hor_pass = ndi.gaussian_filter1d(gray, sigma=sigma, axis = 1)
    ver_pass = ndi.gaussian_filter1d(hor_pass, sigma=sigma, axis = 0)


t2 = time()


h,w = gray.shape

gausses = []

for s in range(1, 5):
    thng = [make_filts(s) for i in range(2 * h + 2 * w)]
    gausses.append(thng)

gausi = List(List(x) for x in gausses)

t3 = time()

for i in range(1):
    img = cv2.imread("input.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = filt(gray, gausi)



t4 = time()

print(len(blurred))

print("TIMES: ", t1-t0, t2-t1, t3-t2, t4-t3, np.sum(np.abs(blurred[0] - stock_gauss)))



cv2.imshow('SIFT Keypoints', (blurred[0] - stock_gauss).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
