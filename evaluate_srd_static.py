import cv2
import srd_sift as srd_sift
from skimage.feature import SIFT, match_descriptors, plot_matched_features
import radial as radial
import matplotlib.pyplot as plt
from matching import Keypoints, D_MATCHER
import numpy as np
from joblib import Parallel, delayed
from water_surface_simulator import WaterSurfaceSimulator
import numpy as np
from typing import Optional, Tuple, Literal
from hpatches_images import image_list
from tqdm import tqdm

# h = 80
# d = 20
f = 1
A = 1.0 / 1.333

def eval(gray, amp, h, d):
    # Load an image to evaluate
    # img = cv2.imread(image)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Store the statistics of a given run
    results = np.zeros(7)
    
    # Create a water simulator with a distribution of waves of amplitude around amp
    sim = WaterSurfaceSimulator(gray, h = h, d = d, amplitude_range=(0.7 * amp, 1.3 * amp))
    if (amp == 0):
        WaterSurfaceSimulator(gray, n_waves=0)

    frames = sim.generate_frames()
    dist_gray = frames[0]

    dist_func = sim.make_distortion_func(t=0)


    xi = d * A * (1 - A**2) / (2 * (h + d))
    desc_rd = srd_sift.SIFT(use_jacobian_correction=True)
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
ns = np.linspace(1, 5, 10)
#xis = np.linspace(0.055, 0.167, 10)
#amps = [0.0002]


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
        np.savetxt(f"static-{angle}-{im}-results", results)
    im += 1

#for image in tqdm(image_list()[0]):
# for run in range(1):
# results = eval(image_list()[0], amps[0])#Parallel(n_jobs=-1, verbose = 15)(delayed(eval)(image_list()[0], amp) for amp in amps)
# results = np.array(results).T
# print(results)
# np.savetxt(f"image-1-static", results)

# plt.plot(results[6], results[0], color = 'blue')
# plt.plot(results[6], results[3], color = 'red')

# plt.show()

# idx1 = np.argsort(results[2])
# idx2 = np.argsort(results[5])

# plt.plot(results[2][idx1], results[1][idx1], color = 'green')
# plt.plot(results[5][idx1], results[4][idx1], color = 'orange')

# plt.show()



