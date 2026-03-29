import cv2
import numpy as np
from srd_sift import SIFT as srd_sift
from skimage.feature import SIFT, match_descriptors


rd_desc = srd_sift()
desc = SIFT()

xi = -0.5

image = cv2.imread("input.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rd_desc._create_1d_water_gaussians(gray.shape, h=1, d=1, A = 0.75)
print("done")
(rd_desc._create_jacobians(gray.shape, xi))
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

rd_desc.detect_and_extract(gray, 1)
# desc.detect_and_extract(gray)

# matches = match_descriptors(descriptors1=desc.descriptors,
#                                             descriptors2=rd_desc.descriptors,
#                                             cross_check=True,)

# print("Ratio:", len(matches) / len(desc.keypoints))