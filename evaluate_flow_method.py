from dataclasses import dataclass
from typing import List, Optional
from water_surface_simulator import WaterSurfaceSimulator
from skimage.feature import SIFT
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matching import Keypoints, D_MATCHER
import cv2
import numpy as np
from keypoint_flow import FBFlowTrackResult, ForwardBackwardFlowFilter
from tqdm import tqdm
from srd_sift import SIFT as srd_sift
from joblib import Parallel, delayed
from hpatches_images import image_list


# h = 1
# d = 1
f = 1
A = 1.0 / 1.333


def eval(gray, amp, h, d):

    # Store the statistics of a given run
    results = np.zeros(18)
    
    # Create a water simulator with a distribution of waves of amplitude around amp
    sim = WaterSurfaceSimulator(gray, h = h, d = d, amplitude_range=(0.7 * amp, 1.3 * amp))
    if (amp == 0):
        WaterSurfaceSimulator(gray, n_waves=0)

    frames = sim.generate_frames()
    dist_gray = frames[0]

    dist_func = sim.make_distortion_func(t=0)


    xi = d * A * (1 - A**2) / (2 * (h + d))
    desc_rd = srd_sift(xi = xi)
    # desc_rd._create_1d_gaussians(dist_gray.shape, xi)
    # (desc_rd._create_jacobians(gray.shape, xi))
    desc_rd.detect_and_extract(dist_gray, 1)

    keypoints1 = desc_rd.keypoints
    descriptors1 = desc_rd.descriptors
    scales1 = desc_rd.sigmas
    distorted_kps = Keypoints(keypoints1[:, ::-1], descriptors1, scales1)

    keypoints_flow1 = keypoints1[:, ::-1].astype(np.float32)

    fb_filter = ForwardBackwardFlowFilter(
        win_size=(21, 21),
        max_level=3,
        fb_error_threshold=1.5,
        min_successful_steps=len(frames) - 1,
    )

    out = fb_filter.filter_points(frames, keypoints_flow1)

    survivor_points0 = out["survivor_points0"]
    survivor_mask = out["survivor_mask"]
    # tracks = out["tracks"]
    # mean_fb_errors = out["mean_fb_errors"]

    kp_flow1 = keypoints1[survivor_mask]
    desc_flow1 = descriptors1[survivor_mask]
    scales_flow1 = scales1[survivor_mask]
    distorted_flow_kps = Keypoints(kp_flow1[:, ::-1], desc_flow1, scales_flow1)


    desc = SIFT()
    desc.detect_and_extract(gray)
    keypoints2 = desc.keypoints
    descriptors2 = desc.descriptors
    scales2 = desc.sigmas
    origin_kps = Keypoints(keypoints2[:, ::-1], descriptors2, scales2)

    desc.detect_and_extract(dist_gray)
    keypoints3 = desc.keypoints
    descriptors3 = desc.descriptors
    scales3 = desc.sigmas
    sift_kps = Keypoints(keypoints3[:, ::-1], descriptors3, scales3)



    keypoints_flow3 = keypoints3[:, ::-1].astype(np.float32)

    fb_filter = ForwardBackwardFlowFilter(
        win_size=(21, 21),
        max_level=3,
        fb_error_threshold=1.5,
        min_successful_steps=len(frames) - 1,
    )

    out = fb_filter.filter_points(frames, keypoints_flow3)

    survivor_points0 = out["survivor_points0"]
    survivor_mask = out["survivor_mask"]

    kp_flow3 = keypoints3[survivor_mask]
    desc_flow3 = descriptors3[survivor_mask]
    scales_flow3 = scales3[survivor_mask]
    sift_flow_kps = Keypoints(kp_flow3[:, ::-1], desc_flow3, scales_flow3)


    if (len(keypoints2) == 0 or len(keypoints1) == 0):
        pass
    else:
        matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=distorted_kps, distortion_func=dist_func)
        out = matcher.compute_stats()
        results[0] = out['repeatability']
        results[1] = out['recall']
        results[2] = out['precision']

    if (len(keypoints2) == 0 or len(kp_flow1) == 0):
        pass
    else:
        matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=distorted_flow_kps, distortion_func=dist_func)
        out = matcher.compute_stats()
        results[3] = out['repeatability']
        results[4] = out['recall']
        results[5] = out['precision']

    if (len(keypoints2) == 0 or len(keypoints3) == 0):
        pass
    else:
        matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=sift_kps, distortion_func=dist_func)
        out = matcher.compute_stats()
        results[6] = out['repeatability']
        results[7] = out['recall']
        results[8] = out['precision']


    if (len(keypoints2) == 0 or len(kp_flow3) == 0):
        pass
    else:
        matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=sift_flow_kps, distortion_func=dist_func)
        out = matcher.compute_stats()
        results[9] = out['repeatability']
        results[10] = out['recall']
        results[11] = out['precision']

    results[12] = len(keypoints2)
    results[13] = len(keypoints1)
    results[14] = len(kp_flow1)
    results[15] = len(keypoints3)
    results[16] = len(kp_flow3)

    results[17] = amp

    return results



amps = np.linspace(0.0, 0.002, 10)
angles = np.linspace(0.01, 1, 10)
#amps = [0.0002]
im = 0
print(len(image_list()))
for image in tqdm(image_list()):
    for angle in angles:
        h = 1
        d = h * angle
        results = Parallel(n_jobs=-1, verbose = 15)(delayed(eval)(image, amp, h=h, d = d) for amp in amps)
        results = np.array(results).T
        np.savetxt(f"flow-{im}-{angle}-results", results)
    im += 1



