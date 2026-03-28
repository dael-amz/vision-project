import cv2
import srd_sift as srd_sift
from skimage.feature import SIFT, match_descriptors, plot_matched_features
import radial
import matplotlib.pyplot as plt
from matching import Keypoints, D_MATCHER

img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

xi = -0.1

dist_gray, _, _ = radial.generate_distorted_image(gray, xi)

cv2.imshow('SIFT Keypoints', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np


import numpy as np


def make_division_distortion_func(
    xi,
    image_shape,
    center=None,
    norm_scale=None,
    scale_mode="radial",
):
    """
    Build distortion_func(origin_loc, origin_scales) that matches
    generate_distorted_image(...).

    Parameters
    ----------
    xi : float
        Division-model parameter.
    image_shape : tuple
        Image shape, e.g. gray.shape
    center : tuple or None
        Distortion center. If None, uses image center exactly like generate_distorted_image.
    norm_scale : float or None
        Coordinate normalization scale. If None, uses max(h, w) * 0.5,
        exactly like generate_distorted_image.
    scale_mode : str
        "radial", "area", or "none"
    """
    h, w = image_shape[:2]

    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    if norm_scale is None:
        norm_scale = max(h, w) * 0.5

    cx, cy = center

    def distortion_func(origin_loc, origin_scales):
        origin_loc = np.asarray(origin_loc, dtype=float)
        origin_scales = np.asarray(origin_scales, dtype=float)

        # Normalize exactly like generate_distorted_image
        xu = (origin_loc[:, 0] - cx) / norm_scale
        yu = (origin_loc[:, 1] - cy) / norm_scale
        ru2 = xu * xu + yu * yu

        inside = 1.0 - 4.0 * xi * ru2
        sqrt_inside = np.sqrt(inside)

        # Forward map: undistorted -> distorted
        scale = 2.0 / (1.0 + sqrt_inside)
        xd = xu * scale
        yd = yu * scale

        distorted_loc = np.column_stack([
            xd * norm_scale + cx,
            yd * norm_scale + cy
        ])

        # Scalar scale mapping
        if scale_mode == "none":
            distorted_scales = origin_scales.copy()

        elif scale_mode == "radial":
            # r_d = r_u * scale
            # scale = 2 / (1 + sqrt(1 - 4 xi r_u^2))
            # dr_d/dr_u = 2 / (sqrt_inside * (1 + sqrt_inside))
            mag = 2.0 / (sqrt_inside * (1.0 + sqrt_inside))
            distorted_scales = origin_scales * mag

        elif scale_mode == "area":
            tangential_mag = scale
            radial_mag = 2.0 / (sqrt_inside * (1.0 + sqrt_inside))
            area_mag = np.sqrt(np.abs(tangential_mag * radial_mag))
            distorted_scales = origin_scales * area_mag

        else:
            raise ValueError(f"Unknown scale_mode: {scale_mode}")

        return distorted_loc, distorted_scales

    return distortion_func


desc_rd = srd_sift.SIFT()
#xi = 0#-0.5 / 1000 * (gray.shape[0]**2 +gray.shape[1]**2)
desc_rd._create_1d_gaussians(dist_gray.shape, xi)
desc_rd.detect_and_extract(dist_gray, 1)

keypoints1 = desc_rd.keypoints
descriptors1 = desc_rd.descriptors
scales1 = desc_rd.scales
orientations1 = desc_rd.orientations
distorted_kps = Keypoints(keypoints1[:, ::-1], descriptors1, scales1)

desc = SIFT()
desc.detect_and_extract(gray)
keypoints2 = desc.keypoints
descriptors2 = desc.descriptors
scales2 = desc.scales
origin_kps = Keypoints(keypoints2[:, ::-1], descriptors2, scales2)

dist_func = make_division_distortion_func(xi=xi, image_shape=gray.shape)

matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=distorted_kps, scale_thresh=0.5, pos_thresh=3.0, distortion_func=dist_func)
out = matcher.compute_stats()
print(out)