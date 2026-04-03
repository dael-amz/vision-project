import cv2
import srd_sift as srd_sift
from skimage.feature import SIFT, match_descriptors, plot_matched_features
import radial as radial
import matplotlib.pyplot as plt
from matching import Keypoints, D_MATCHER
import numpy as np
from joblib import Parallel, delayed
from water_surface_simulator import WaterSurfaceSimulator

img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# dist_r = np.max(dist_gray.shape)
# r = np.max(gray.shape)

# print("Distortion: ", 100 * (r - dist_r) / r)

# cv2.imshow('SIFT Keypoints', dist_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import numpy as np


# import numpy as np


import numpy as np
from typing import Optional, Tuple, Literal


def make_division_distortion_func(
    xi: float,
    image_shape: Tuple[int, int],
    center: Optional[Tuple[float, float]] = None,
    norm_scale: Optional[float] = None,
    scale_mode: Literal["none", "srd"] = "srd",
):
    """
    Build a function that maps undistorted keypoints to distorted keypoints
    under the same normalized division model used in the paper.

    Forward model:
        x = f(u)
          = 2u / (1 + sqrt(1 - 4 xi ||u||^2))

    This is the paper's Eq. (3). The generator above uses Eq. (4) because
    image synthesis needs backward warping, while evaluation of point
    correspondences usually needs the forward map. :contentReference[oaicite:1]{index=1}
    """
    h, w = image_shape[:2]

    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    if norm_scale is None:
        norm_scale = max(h, w) * 0.5

    cx, cy = center
    s = float(norm_scale)

    def distortion_func(origin_loc, origin_scales):
        origin_loc = np.asarray(origin_loc, dtype=np.float64)
        origin_scales = np.asarray(origin_scales, dtype=np.float64)

        # origin_loc assumed in (x, y) = (col, row) pixel convention
        uu = (origin_loc[:, 0] - cx) / s
        vv = (origin_loc[:, 1] - cy) / s
        ru2 = uu * uu + vv * vv

        inside = 1.0 - 4.0 * xi * ru2
        inside = np.maximum(inside, 0.0)
        scale = 2.0 / (1.0 + np.sqrt(inside))

        xd = uu * scale
        yd = vv * scale

        distorted_loc = np.column_stack([
            xd * s + cx,
            yd * s + cy,
        ])

        if scale_mode == "none":
            distorted_scales = origin_scales.copy()
        elif scale_mode == "srd":
            # linear local scale factor used by the paper for repeatability correction
            distorted_scales = origin_scales * (1.0 + xi * ru2)
        else:
            raise ValueError(f"Unknown scale_mode: {scale_mode}")

        return distorted_loc, distorted_scales

    return distortion_func


h, w = gray.shape
norm_scale = max(h, w) / 2
rad = np.sqrt((h / 2)**2 + (w / 2)**2)


def eval(xi):
    #xi = -0.109
    results = np.zeros(7)

    # H = 1.0
    # d = 1.0
    # n_water = 1.3333
    # n_air = 1.0
    # A = n_air / n_water
    
    # sim = WaterSurfaceSimulator(gray,h=H, d=d, f=1, amplitude_range=(0.002 * amp, 0.02 * amp))
    # if (amp == 0):
    #     WaterSurfaceSimulator(gray, n_waves=0)

    effecitve_distortion =  - 100 * xi * (rad / norm_scale)**2
    dist_gray, _, _ = radial.generate_distorted_image(gray, xi)
    # frames = sim.generate_frames()
    # dist_gray = frames[10]
    # time = 0.04 * 10

    #dist_func = sim.make_distortion_func(t=time)
    print("Progress 1")

    desc_rd = srd_sift.SIFT()
    desc_rd._create_1d_gaussians(dist_gray.shape, xi=xi)
    (desc_rd._create_jacobians(gray.shape, xi))
    desc_rd.detect_and_extract(dist_gray, 1)

    keypoints1 = desc_rd.keypoints
    descriptors1 = desc_rd.descriptors
    scales1 = desc_rd.sigmas
    orientations1 = desc_rd.orientations
    print(np.unique(scales1))
    distorted_kps = Keypoints(keypoints1[:, ::-1], descriptors1, scales1)
    print("Progress 2")

    desc = SIFT()
    desc.detect_and_extract(gray)
    keypoints2 = desc.keypoints
    descriptors2 = desc.descriptors
    scales2 = desc.sigmas
    print(np.unique(scales2))
    origin_kps = Keypoints(keypoints2[:, ::-1], descriptors2, scales2)
    print("Progress 3")

    desc.detect_and_extract(dist_gray)
    keypoints3 = desc.keypoints
    descriptors3 = desc.descriptors
    scales3 = desc.sigmas
    print(scales3)
    s_kps = Keypoints(keypoints3[:, ::-1], descriptors3, scales3)
    print("Progress 4")

    dist_func = make_division_distortion_func(xi=xi, image_shape=gray.shape)

    matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=distorted_kps, distortion_func=dist_func)
    out = matcher.compute_stats()
    results[0] = out['repeatability']
    results[1] = out['recall']
    results[2] = out['precision']

    matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=s_kps, distortion_func=dist_func)
    out = matcher.compute_stats()
    results[3] = out['repeatability']
    results[4] = out['recall']
    results[5] = out['precision']

    results[6] = effecitve_distortion

    return results



# vals = np.arange(0, 90, 10)
# xis = vals / (- 100 * (rad / norm_scale)**2)

# # amps = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# # results = np.zeros((2, len(vals)))

# # i = 0

# results = eval(xis[1])#Parallel(n_jobs=-1)(delayed(eval)(amp) for amp in amps)
# results = np.array(results).T
# print("rResults", results.T)

# plt.plot(results[6], results[0], label = f'sRD-SIFT ')
# plt.plot(results[6], results[3], label = f'SIFT ')
# plt.xlabel('distortion')
# plt.ylabel('repeatability')
# plt.title('sRD-SIFT vs SIFT')
# plt.legend()
# print("Progress 5")

# plt.show()
