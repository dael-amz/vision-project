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


def eval(image, amp):
    # Load an image to evaluate
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Store the statistics of a given run
    results = np.zeros(7)
    
    # Create a water simulator with a distribution of waves of amplitude around amp
    sim = WaterSurfaceSimulator(gray, amplitude_range=(0.7 * amp, 1.3 * amp))
    if (amp == 0):
        WaterSurfaceSimulator(gray, n_waves=0)

    frames = sim.generate_frames()
    dist_gray = frames[0]

    dist_func = sim.make_distortion_func(t=0)


    xi = -0.1
    desc_rd = srd_sift.SIFT()
    desc_rd._create_1d_gaussians(dist_gray.shape, xi)
    (desc_rd._create_jacobians(gray.shape, xi))
    desc_rd.detect_and_extract(dist_gray, 1)

    keypoints1 = desc_rd.keypoints
    descriptors1 = desc_rd.descriptors
    scales1 = desc_rd.sigmas
    distorted_kps = Keypoints(keypoints1[:, ::-1], descriptors1, scales1)

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

    matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=distorted_kps, distortion_func=dist_func)
    out = matcher.compute_stats()
    results[0] = out['repeatability']
    results[1] = out['recall']
    results[2] = out['precision']

    matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=sift_kps, distortion_func=dist_func)
    out = matcher.compute_stats()
    results[3] = out['repeatability']
    results[4] = out['recall']
    results[5] = out['precision']

    results[6] = amp

    return results



amps = np.linspace(0.00002, 0.002, 10)


for image in images:
    for run in range(5):
        results = Parallel(n_jobs=-1)(delayed(eval)(image, amp) for amp in amps)
        results = np.array(results).T
        np.savetxt(f"{image}-{run}", results)

