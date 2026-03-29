import numpy as np
import cv2

def simulate_flat_water(
    src,                 # submerged picture / texture
    h,                   # camera height above water
    d,                   # depth of picture below water
    f,                   # focal length in pixels
    n_air=1.0,
    n_water=1.333,
    cx=None,
    cy=None,
    plane_scale=1.0      # world-units per source pixel on the plane
):
    """
    src is the image painted on the underwater plane z = -d.
    The output is the camera image seen through a flat air-water interface.

    plane_scale: size of one source pixel in world units on the underwater plane.
                 For example, if 1 source pixel = 0.5 mm, then plane_scale = 0.0005
                 in meters.
    """
    H, W = src.shape[:2]

    if cx is None:
        cx = (W - 1) * 0.5
    if cy is None:
        cy = (H - 1) * 0.5

    eta = n_air / n_water

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # normalized image coordinates
    x = (xx - cx) / f
    y = (yy - cy) / f
    rho2 = x*x + y*y

    # flat-water Snell mapping
    scale = h + d * eta / np.sqrt(1.0 + (1.0 - eta**2) * rho2)

    X = x * scale
    Y = y * scale

    # convert plane coordinates (X,Y) back to source-image pixel coordinates
    # assume src is centered at world point (0,0) on the underwater plane
    map_x = X / plane_scale + cx
    map_y = Y / plane_scale + cy

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    distorted = cv2.remap(
        src,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return distorted

img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

xi = -0.1
h = 20
d = 2
f = h

dist_gray = simulate_flat_water(gray, h, d, f, n_water=1.33)

print(np.max(dist_gray-gray), np.mean(np.abs(dist_gray-gray)))
cv2.imshow('SIFT Keypoints', dist_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = gray[::2, ::2]

dist_gray = simulate_flat_water(gray, h, d, f)

cv2.imshow('SIFT Keypoints', dist_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()