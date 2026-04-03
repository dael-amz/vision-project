import cv2
import numpy as np
from srd_sift import SIFT as srd_sift
from skimage.feature import SIFT, match_descriptors, plot_matched_features
import matplotlib.pyplot as plt
from radial import generate_distorted_image, make_division_distortion_func
from matching import Keypoints, D_MATCHER

xi = -0.9



image = cv2.imread("input.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

h,w = gray.shape
norm_scale = max(h, w) // 2
rad = np.sqrt((h/2)**2 + (w / 2)**2)

xi = -90 / (100.0 * (rad / norm_scale) ** 2)
dist_gray, _, _ = generate_distorted_image(gray, xi)

print(xi)

rd_desc = srd_sift(xi=xi, use_jacobian_correction=True)

# rd_desc._create_1d_gaussians(dist_gray.shape, xi=xi)
# (rd_desc._create_jacobians(dist_gray.shape, xi))
print("Detecting: ")
rd_desc.detect_and_extract(dist_gray, 1)
kp1 = rd_desc.keypoints
d1 = rd_desc.descriptors
sig1 = rd_desc.sigmas
p1 = Keypoints(
        kp1[:, ::-1],   # (row,col) -> (x,y)
        d1,
        sig1
    )



desc = SIFT()
desc.detect_and_extract(gray)
kp2 = desc.keypoints
d2 = desc.descriptors
sig2 = desc.sigmas
p2 = Keypoints(
        kp2[:, ::-1],   # (row,col) -> (x,y)
        d2,
        sig2
    )


matches12 = match_descriptors(d1, d2)


desc2 = SIFT()
desc2.detect_and_extract(dist_gray)
kp3 = desc2.keypoints
d3 = desc2.descriptors
sig3 = desc2.sigmas
p3 = Keypoints(
        kp3[:, ::-1],   # (row,col) -> (x,y)
        d3,
        sig3
    )



dist_func = make_division_distortion_func(xi=xi, image_shape=gray.shape, scale_mode='none')
matcher = D_MATCHER(
    origin_kps=p2,
    distorted_kps=p1,
    distortion_func=dist_func,
)

dist_func = make_division_distortion_func(xi=xi, image_shape=gray.shape, scale_mode='srd')
matcher2 = D_MATCHER(
    origin_kps=p2,
    distorted_kps=p3,
    distortion_func=dist_func,
)

print(matcher.compute_stats(), matcher2.compute_stats())


matches13 = match_descriptors(d2, d3)

clipped12 = matches12[::100]
clipped13 = matches13[::100]

print(kp1[~clipped12[:, 0]])

print(len(matches12), len(matches13), len(kp1), len(kp2), len(kp3))

fig, ax = plt.subplots(nrows=2, figsize=(11, 8))

plt.gray()

plot_matched_features(
    dist_gray,
    gray,
    keypoints0=kp1,
    keypoints1=kp2,
    matches=clipped12,
    ax=ax[0],
)
ax[0].axis('off')
ax[0].set_title("Original Image vs. Flipped Image\n" "(all keypoints and matches)")

plot_matched_features(
    gray,
    dist_gray,
    keypoints0=kp2,
    keypoints1=kp3,
    matches=clipped13,
    ax=ax[1],
)
ax[1].axis('off')
ax[1].set_title(
    "Original Image vs. Transformed Image\n" "(all keypoints and matches)"
)

plt.show()

#rd_scale = rd_desc._create_rd_scalespace(gray)

# scale = desc._create_scalespace(gray)

# for o in range(len(scale)):
#     print(np.max(np.abs(scale[o] - rd_scale[o])))
#     print(scale[o].shape)

# for s in range(6):
#     diff = scale[0][:,:,s] - rd_scale[0][:,:,s]
#     print(s, np.mean(np.abs(diff)), np.max(np.abs(diff)))

# diff = np.abs(scale[0][:, :, 4] - rd_scale[0][:, :, 4])
# print(diff.min(), diff.max(), np.mean(np.abs(diff)))
# vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
# cv2.imshow("abs diff", vis.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#rd_desc.detect_and_extract(gray, 1)
# desc.detect_and_extract(gray)

# matches = match_descriptors(descriptors1=desc.descriptors,
#                                             descriptors2=rd_desc.descriptors,
#                                             cross_check=True,)

# print("Ratio:", len(matches) / len(desc.keypoints))