import cv2
import sift
from skimage.feature import SIFT, match_descriptors, plot_matched_features
import radial
import matplotlib.pyplot as plt
from matching import Keypoints, D_MATCHER

img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dist_gray, _, _ = radial.generate_distorted_image(gray, -1)

cv2.imshow('SIFT Keypoints', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

desc_rd = SIFT()
xi = -1#-0.5 / 1000 * (gray.shape[0]**2 +gray.shape[1]**2)
#desc_rd._create_1d_gaussians(dist_gray.shape, xi)
desc_rd.detect_and_extract(gray)

keypoints1 = desc_rd.keypoints
descriptors1 = desc_rd.descriptors
scales1 = desc_rd.scales
orientations1 = desc_rd.orientations
distorted_kps = Keypoints(keypoints1, descriptors1, scales1)

desc = SIFT()
desc.detect_and_extract(gray)
keypoints2 = desc.keypoints
descriptors2 = desc.descriptors
scales2 = desc.scales
origin_kps = Keypoints(keypoints2, descriptors2, scales2)

matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=distorted_kps, scale_thresh=0.5, pos_thresh=3.0)
out = matcher.compute_stats()
print(out)

# # Match the descriptors using a ratio test
# matches12 = match_descriptors(
#     descriptors1, descriptors2,
#     max_ratio=0.6,
#     cross_check=True # Ensures matches are consistent in both directions
# )

# # Plot the matches
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 8))
# plt.gray()
# plot_matched_features(ax=ax, image0=gray, image1=gray, keypoints0=keypoints1, keypoints1=keypoints2, matches=matches12)
# ax.axis('off')
# ax.set_title("SIFT Keypoint Matches")
# plt.show()


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


# img_with_keypoints = cv2.drawKeypoints(
#     image=dist_gray, 
#     keypoints=cv2_keypoints, 
#     outImage=None, 
#     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
#     color=(0,0,255)
# )


# descriptor_extractor = sift.SIFT()

# descriptor_extractor.detect_and_extract(gray, 0)
# keypoints2 = descriptor_extractor.keypoints
# descriptors2 = descriptor_extractor.descriptors
# scales2 = descriptor_extractor.scales
# orientations2 = descriptor_extractor.orientations

# cv2_keypoints2 = []
#     # skimage keypoints are typically (row, col) coordinates
#     # cv2 keypoints are (x, y) coordinates, so (col, row)
# for i in range(keypoints2.shape[0]):
#     # Assuming kp is a coordinate tuple/array (row, col)
#     # We need to provide dummy/default values for other attributes like size, angle, etc.
#     # size (meaningful neighborhood size) and angle (orientation) are important for "rich" drawing
#     # If skimage provides more info, use it. Here we use defaults/placeholders.
#     kp = keypoints2[i]
#     s = scales2[i]
#     o = orientations2[i]
#     x, y = float(kp[1]), float(kp[0])
#     size = 2 * float(s) # Default size placeholder (adjust as needed)
#     angle = float(o) # Default angle placeholder (OpenCV might handle -1 as no orientation)
#     response = 0
#     octave = 0
#     class_id = -1
#     cv2_keypoints2.append(cv2.KeyPoint(x, y, size, angle, response, octave, class_id))


# img_with_keypoints = cv2.drawKeypoints(
#     image=gray, 
#     keypoints=cv2_keypoints2, 
#     outImage=None, 
#     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
#     color=(0,255,0)
# )

# #img_with_keypoints, _, _ = radial.generate_distorted_image(img_with_keypoints, -0.5)

# img_with_keypoints = cv2.drawKeypoints(
#     image=img_with_keypoints, 
#     keypoints=cv2_keypoints, 
#     outImage=None, 
#     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
#     color=(0,0,255)
# )

# # sift = cv2.SIFT_create()
# # kp, pt = sift.detectAndCompute(gray ,None)

# # print(len(keypoints2), len(kp))

# # img_with_keypoints = cv2.drawKeypoints(
# #     image=img_with_keypoints, 
# #     keypoints=kp, 
# #     outImage=None, 
# #     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
# #     color=(255,0,0)
# # )

# # 5. Display the image
# cv2.imshow('SIFT Keypoints', img_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()