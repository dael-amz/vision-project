import cv2
import numpy as np
import matplotlib.pyplot as plt

import radial as radial
import srd_sift as srd_sift_mod
from skimage.feature import SIFT as StockSIFT
from matching import Keypoints, D_MATCHER
from radial import make_division_distortion_func
import hpatches_images


def extract_stock(gray):
    desc = StockSIFT()
    desc.detect_and_extract(gray)
    return Keypoints(
        desc.keypoints[:, ::-1],   # (row,col) -> (x,y)
        desc.descriptors,
        desc.sigmas
    )


def extract_srd(gray, xi, use_jacobian_correction=True):
    desc = srd_sift_mod.SIFT(xi = xi)
    desc.use_jacobian_correction = use_jacobian_correction

    # desc._create_1d_gaussians(gray.shape, xi)
    # desc._create_jacobians(gray.shape, xi)
    desc.detect_and_extract(gray, 1)

    return Keypoints(
        desc.keypoints[:, ::-1],   # (row,col) -> (x,y)
        desc.descriptors,
        desc.sigmas
    )


def evaluate_pair(origin_kps, distorted_kps, xi, image_shape, mode):
    dist_func = make_division_distortion_func(xi=xi, image_shape=image_shape, scale_mode=mode)
    matcher = D_MATCHER(
        origin_kps=origin_kps,
        distorted_kps=distorted_kps,
        distortion_func=dist_func,
    )
    return matcher.compute_stats()


def run_ablation(image_path="input.jpg", vals=np.arange(-90, 90, 10)):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = hpatches_images.image_list()[0]

    cv2.imshow("New Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    h, w = gray.shape
    norm_scale = max(h, w) / 2
    rad = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)

    xis = vals / (100.0 * (rad / norm_scale) ** 2)

    origin_kps = extract_stock(gray)

    out = {
        "distortion_pct": [],
        "stock_repeatability": [],
        "stock_recall": [],
        "stock_precision": [],
        "srd_noJ_repeatability": [],
        "srd_noJ_recall": [],
        "srd_noJ_precision": [],
        "srd_J_repeatability": [],
        "srd_J_recall": [],
        "srd_J_precision": [],
    }

    for xi in xis:
        print(f"Testing {xi}")
        distorted_gray, _, _ = radial.generate_distorted_image(gray, xi)
        

        # stock SIFT on distorted image
        stock_dist_kps = extract_stock(distorted_gray)

        # sRD-SIFT without Jacobian correction
        srd_noJ_kps = extract_srd(
            distorted_gray, xi, use_jacobian_correction=False
        )

        # sRD-SIFT with Jacobian correction
        srd_J_kps = extract_srd(
            distorted_gray, xi, use_jacobian_correction=True
        )

        stock_stats = evaluate_pair(origin_kps, stock_dist_kps, xi, gray.shape, mode='srd')
        srd_noJ_stats = evaluate_pair(origin_kps, srd_noJ_kps, xi, gray.shape, mode='srd')
        srd_J_stats = evaluate_pair(origin_kps, srd_J_kps, xi, gray.shape, mode='srd')

        distortion_pct = -100.0 * xi * (rad / norm_scale) ** 2
        out["distortion_pct"].append(distortion_pct)

        out["stock_repeatability"].append(stock_stats["repeatability"])
        out["stock_recall"].append(stock_stats["recall"])
        out["stock_precision"].append(stock_stats["precision"])

        out["srd_noJ_repeatability"].append(srd_noJ_stats["repeatability"])
        out["srd_noJ_recall"].append(srd_noJ_stats["recall"])
        out["srd_noJ_precision"].append(srd_noJ_stats["precision"])

        out["srd_J_repeatability"].append(srd_J_stats["repeatability"])
        out["srd_J_recall"].append(srd_J_stats["recall"])
        out["srd_J_precision"].append(srd_J_stats["precision"])

        print(f"xi={xi:.5f}, distortion={distortion_pct:.1f}%")
        print("  stock   :", stock_stats)
        print("  srd_noJ :", srd_noJ_stats)
        print("  srd_J   :", srd_J_stats)
        print()

    return out


def plot_results(out):
    x = np.array(out["distortion_pct"])

    plt.figure()
    plt.plot(x, out["stock_repeatability"], label="Stock SIFT")
    plt.plot(x, out["srd_noJ_repeatability"], label="sRD-SIFT no Jacobian")
    plt.plot(x, out["srd_J_repeatability"], label="sRD-SIFT + Jacobian")
    plt.xlabel("effective distortion (%)")
    plt.ylabel("repeatability")
    plt.title("Repeatability vs distortion")
    plt.legend()

    plt.figure()
    plt.plot(x, out["stock_recall"], label="Stock SIFT")
    plt.plot(x, out["srd_noJ_recall"], label="sRD-SIFT no Jacobian")
    plt.plot(x, out["srd_J_recall"], label="sRD-SIFT + Jacobian")
    plt.xlabel("effective distortion (%)")
    plt.ylabel("recall")
    plt.title("Recall vs distortion")
    plt.legend()

    plt.figure()
    plt.plot(x, out["stock_precision"], label="Stock SIFT")
    plt.plot(x, out["srd_noJ_precision"], label="sRD-SIFT no Jacobian")
    plt.plot(x, out["srd_J_precision"], label="sRD-SIFT + Jacobian")
    plt.xlabel("effective distortion (%)")
    plt.ylabel("precision")
    plt.title("Precision vs distortion")
    plt.legend()

    # direct gain from Jacobian correction
    plt.figure()
    gain = np.array(out["srd_J_recall"]) - np.array(out["srd_noJ_recall"])
    plt.plot(x, gain, label="Recall gain from Jacobian")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("effective distortion (%)")
    plt.ylabel("delta recall")
    plt.title("Jacobian correction gain")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    print("here")
    out = run_ablation("input.jpg")
    plot_results(out)