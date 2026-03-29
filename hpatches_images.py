import cv2
import os


def image_list():
    seq_dir = "hpatches-sequences-release/i_ajuntament"

    images = []
    for i in range(1, 7):
        path = os.path.join(seq_dir, f"{i}.ppm")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images.append(img)

    seq_dir = "hpatches-sequences-release/i_ski"

    for i in range(1, 7):
        path = os.path.join(seq_dir, f"{i}.ppm")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return images
