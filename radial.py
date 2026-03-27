import numpy as np
import cv2

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from sift import SIFT

Array = np.ndarray


class DistortionModel:
    """
    Division-model radial distortion used by the sRD-SIFT paper.

    Coordinates are assumed to be normalized around `center` by `norm_scale`:
        x_n = (x - cx) / norm_scale
        y_n = (y - cy) / norm_scale

    The forward mapping used in the paper is x = u / (1 + xi * ||u||^2).
    This implementation works directly in distorted-image coordinates and uses
    the closed-form Jacobian reported in the paper for implicit gradient correction.

    IMPORTANT:
      * xi is NOT OpenCV's k1.
      * For barrel distortion in the paper's convention, xi is typically negative.
      * The center can be approximate; the paper reports reasonable robustness.
    """

    xi: float
    center: Tuple[float, float]
    image: Array
    radii: Array
    norm_scale: float

    def __init__(self, xi, center):
        self.xi = xi
        self.center = center

    def load_image(self, img_path):
        self.image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    def from_image_shape(self):
        h, w = self.image.shape[:2]
        if self.center is None:
            self.center = ((w - 1) * 0.5, (h - 1) * 0.5)
        if norm_scale is None:
            norm_scale = max(h, w) * 0.5
        radii = np.empty_like(self.image)
        return 

    def normalize_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        cx, cy = self.center
        s = self.norm_scale
        return (x - cx) / s, (y - cy) / s

    def radius2_from_pixels(self, x: Array, y: Array) -> Array:
        xn, yn = self.normalize_points(x, y)
        return xn * xn + yn * yn

    def scale_factor(self, x: Array, y: Array) -> Array:
        """Adaptive Gaussian scale factor 1 + xi r^2 from Eq. (15)."""
        r2 = self.radius2_from_pixels(x, y)
        sf = 1.0 + self.xi * r2
        return np.maximum(sf, 1e-4)

    def jacobian_from_pixels(self, x: Array, y: Array) -> Array:
        """
        Jacobian J_f in distorted-image coordinates as reported in the paper.
        Returns array shape (..., 2, 2).
        """
        xn, yn = self.normalize_points(x, y)
        r2 = xn * xn + yn * yn
        xi = self.xi
        denom = np.maximum(1.0 - xi * r2, 1e-8)
        pref = (1.0 + xi * r2) / denom

        j11 = pref * (1.0 - xi * (r2 - 8.0 * xn * xn))
        j12 = pref * (8.0 * xi * xn * yn)
        j21 = j12
        j22 = pref * (1.0 - xi * (r2 - 8.0 * yn * yn))

        J = np.empty(xn.shape + (2, 2), dtype=np.float32)
        J[..., 0, 0] = j11
        J[..., 0, 1] = j12
        J[..., 1, 0] = j21
        J[..., 1, 1] = j22
        return J


def generate_distorted_image(
    image: Array,
    xi: float,
    center: Optional[Tuple[float, float]] = None,
    norm_scale: Optional[float] = None,
    interpolation: int = cv2.INTER_CUBIC,
) -> Tuple[Array, Tuple[float, float], float]:
    h, w = image.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    if norm_scale is None:
        norm_scale = max(h, w) * 0.5

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xd = (xx - center[0]) / norm_scale
    yd = (yy - center[1]) / norm_scale
    rd2 = xd * xd + yd * yd

    inv_scale = 2.0 / (1.0 + np.sqrt(1 - 4 * xi * rd2))
    xu = xd * inv_scale
    yu = yd * inv_scale

    map_x = xu * norm_scale + center[0]
    map_y = yu * norm_scale + center[1]

    distorted = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return distorted, center, float(norm_scale)

# img = cv2.imread("input.jpg")
# test = generate_distorted_image(img, 0.1)[0]
# cv2.imshow('test', test, )
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# descriptor_extractor = SIFT()

# descriptor_extractor.detect_and_extract(gray)
# keypoints1 = descriptor_extractor.keypoints
# descriptors1 = descriptor_extractor.descriptors
# scales1 = descriptor_extractor.scales
# orientations1 = descriptor_extractor.orientations

# test1, test2, test3 = descriptor_extractor._create_1d_gaussians(gray.shape, -0.1)
# print((test2[0].keys()))


# sift = cv2.SIFT_create(sigma = 10)
# kp, pt = sift.detectAndCompute(gray ,None)


# cv2_keypoints = []
#     # skimage keypoints are typically (row, col) coordinates
#     # cv2 keypoints are (x, y) coordinates, so (col, row)
# for i in range(keypoints1.shape[0]):
#     # Assuming kp is a coordinate tuple/array (row, col)
#     # We need to provide dummy/default values for other attributes like size, angle, etc.
#     # size (meaningful neighborhood size) and angle (orientation) are important for "rich" drawing
#     # If skimage provides more info, use it. Here we use defaults/placeholders.
#     kp = keypoints1[i]
#     s = scales1[i]
#     o = orientations1[i]
#     x, y = float(kp[1]), float(kp[0])
#     size = float(s) # Default size placeholder (adjust as needed)
#     angle = float(o) # Default angle placeholder (OpenCV might handle -1 as no orientation)
#     response = 0
#     octave = 0
#     class_id = -1
#     cv2_keypoints.append(cv2.KeyPoint(x, y, size, angle, response, octave, class_id))

# print(kp, keypoints1)

# img_with_keypoints = cv2.drawKeypoints(
#     image=img, 
#     keypoints=cv2_keypoints, 
#     outImage=None, 
#     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )

# # 5. Display the image
# cv2.imshow('SIFT Keypoints', img_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()