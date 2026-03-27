import cv2
from srd_sift_fast import SRDSIFT

img = cv2.imread("input.jpg")
detector = SRDSIFT(max_keypoints=2000)

keypoints, descriptors = detector.detectAndCompute(
    img,
    xi=-0.20,
    center=None,
    norm_scale=None,
)

vis = detector.draw_keypoints(img, keypoints)
cv2.imwrite("srd_sift_keypoints.png", vis)

print(len(keypoints))
print(None if descriptors is None else descriptors.shape)