import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SRDSIFTConfig:
    xi: float                              # division-model radial distortion parameter
    center: Optional[Tuple[float, float]] = None  # distortion center (cx, cy); default = image center
    num_radial_bins: int = 24              # for approximate adaptive blur
    sift_nfeatures: int = 0
    sift_n_octave_layers: int = 3
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0
    sift_sigma: float = 1.6
    descriptor_width: int = 4              # 4x4 cells
    descriptor_bins: int = 8               # 8 orientation bins
    magnification: float = 3.0             # SIFT-like support size
    clip_value: float = 0.2
    eps: float = 1e-8


class SRDSIFT:
    """
    Practical sRD-SIFT implementation:
      1) approximate adaptive blur in the distorted image
      2) detect keypoints with OpenCV SIFT
      3) compute custom 128-D descriptors using Jacobian-corrected gradients
    """

    def __init__(self, config: SRDSIFTConfig):
        self.cfg = config
        self._sift = cv2.SIFT_create(
            nfeatures=config.sift_nfeatures,
            nOctaveLayers=config.sift_n_octave_layers,
            contrastThreshold=config.sift_contrast_threshold,
            edgeThreshold=config.sift_edge_threshold,
            sigma=config.sift_sigma,
        )

    def detect_and_compute(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        gray = self._to_gray_f32(image)
        h, w = gray.shape

        cx, cy = self._center(w, h)

        # 1) Approximate sRD-SIFT adaptive filtering for detection
        adapt = self._adaptive_radial_blur(gray, cx, cy)

        # 2) Detect keypoints on the adaptively blurred image
        keypoints = self._sift.detect((adapt * 255.0).astype(np.uint8), None)

        # 3) Compute gradients on the original distorted image
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # 4) Custom sRD-SIFT descriptors using implicit gradient correction
        descs = []
        valid_kps = []
        for kp in keypoints:
            d = self._compute_descriptor(gray, gx, gy, kp, cx, cy)
            if d is not None:
                valid_kps.append(kp)
                descs.append(d)

        if len(descs) == 0:
            return [], np.zeros((0, 128), dtype=np.float32)

        return valid_kps, np.vstack(descs).astype(np.float32)

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_test: float = 0.75,
    ) -> List[cv2.DMatch]:
        if len(desc1) == 0 or len(desc2) == 0:
            return []
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(desc1, desc2, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < ratio_test * n.distance:
                good.append(m)
        return good

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _to_gray_f32(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = gray.astype(np.float32)
        if gray.max() > 1.0:
            gray /= 255.0
        return gray

    def _center(self, w: int, h: int) -> Tuple[float, float]:
        if self.cfg.center is not None:
            return self.cfg.center
        return (w - 1) * 0.5, (h - 1) * 0.5

    def _normalized_xy(
        self, x: np.ndarray, y: np.ndarray, cx: float, cy: float, w: int, h: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # normalized coordinates around center; scale chosen from image size
        s = 0.5 * max(w, h)
        xn = (x - cx) / s
        yn = (y - cy) / s
        return xn, yn

    def _adaptive_radial_blur(
        self, gray: np.ndarray, cx: float, cy: float
    ) -> np.ndarray:
        """
        Approximate the paper's adaptive Gaussian blur by radial binning:
        each ring gets a different blur sigma:
            sigma_eff(r) = sigma0 * (1 + xi * r^2)
        """
        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w]
        xn, yn = self._normalized_xy(xx, yy, cx, cy, w, h)
        r2 = xn * xn + yn * yn
        r = np.sqrt(r2)

        max_r = max(np.max(r), 1e-6)
        bins = np.linspace(0.0, max_r, self.cfg.num_radial_bins + 1)

        out = np.zeros_like(gray)
        weight_sum = np.zeros_like(gray)

        sigma0 = self.cfg.sift_sigma
        xi = self.cfg.xi

        for i in range(self.cfg.num_radial_bins):
            r0, r1 = bins[i], bins[i + 1]
            rc = 0.5 * (r0 + r1)
            sigma_eff = max(0.01, sigma0 * (1.0 + xi * rc * rc))

            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_eff, sigmaY=sigma_eff)

            mask = ((r >= r0) & (r <= r1)).astype(np.float32)
            out += blurred * mask
            weight_sum += mask

        out /= np.maximum(weight_sum, self.cfg.eps)
        return out

    def _jacobian_corrected_gradient(
        self,
        gx: float,
        gy: float,
        x: float,
        y: float,
        cx: float,
        cy: float,
        w: int,
        h: int,
    ) -> Tuple[float, float]:
        """
        Implicit gradient correction from the paper:
            ∇I_u = J_f ∇I
        using the Jacobian formula quoted in the paper.
        """
        xi = self.cfg.xi
        xn, yn = self._normalized_xy(
            np.array([x], dtype=np.float32),
            np.array([y], dtype=np.float32),
            cx, cy, w, h
        )
        xn = float(xn[0])
        yn = float(yn[0])
        r2 = xn * xn + yn * yn

        denom = 1.0 - xi * r2
        if abs(denom) < 1e-6:
            return gx, gy

        factor = (1.0 + xi * r2) / denom

        j11 = factor * (1.0 - xi * (r2 - 8.0 * xn * xn))
        j12 = factor * (8.0 * xi * xn * yn)
        j21 = j12
        j22 = factor * (1.0 - xi * (r2 - 8.0 * yn * yn))

        gux = j11 * gx + j12 * gy
        guy = j21 * gx + j22 * gy
        return gux, guy

    def _compute_descriptor(
        self,
        gray: np.ndarray,
        gx_img: np.ndarray,
        gy_img: np.ndarray,
        kp: cv2.KeyPoint,
        cx: float,
        cy: float,
    ) -> Optional[np.ndarray]:
        """
        128-D SIFT-like descriptor built from Jacobian-corrected gradients.
        """
        h, w = gray.shape
        x0, y0 = kp.pt
        angle = math.radians(kp.angle if kp.angle >= 0 else 0.0)

        cos_t = math.cos(angle)
        sin_t = math.sin(angle)

        d = self.cfg.descriptor_width
        n = self.cfg.descriptor_bins
        magnif = self.cfg.magnification
        clip_value = self.cfg.clip_value

        # Support window size, roughly SIFT-like.
        # kp.size is diameter-ish in OpenCV; use half for scale proxy.
        scale = max(kp.size * 0.5, 1.0)
        hist_width = magnif * scale
        radius = int(round(math.sqrt(2) * hist_width * (d + 1) * 0.5))

        if (
            x0 < radius or x0 >= (w - radius) or
            y0 < radius or y0 >= (h - radius)
        ):
            return None

        hist = np.zeros((d, d, n), dtype=np.float32)

        # Gaussian spatial weighting; paper also adapts weighting with (1 + xi r^2)
        x0n, y0n = self._normalized_xy(
            np.array([x0], dtype=np.float32),
            np.array([y0], dtype=np.float32),
            cx, cy, w, h
        )
        r2_kp = float(x0n[0] * x0n[0] + y0n[0] * y0n[0])
        sigma_w = 0.5 * d * hist_width * (1.0 + self.cfg.xi * r2_kp)
        sigma_w2 = max(sigma_w * sigma_w, self.cfg.eps)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = int(round(x0 + dx))
                y = int(round(y0 + dy))

                if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
                    continue

                # Rotate sample location into keypoint frame
                rx = (cos_t * dx + sin_t * dy) / hist_width
                ry = (-sin_t * dx + cos_t * dy) / hist_width

                # descriptor cell coordinates
                cx_bin = rx + d / 2 - 0.5
                cy_bin = ry + d / 2 - 0.5

                if cx_bin <= -1 or cx_bin >= d or cy_bin <= -1 or cy_bin >= d:
                    continue

                gx = float(gx_img[y, x])
                gy = float(gy_img[y, x])

                # Implicit gradient correction
                gux, guy = self._jacobian_corrected_gradient(gx, gy, x, y, cx, cy, w, h)

                mag = math.hypot(gux, guy)
                if mag < 1e-12:
                    continue

                ori = math.atan2(guy, gux) - angle
                while ori < 0:
                    ori += 2 * math.pi
                while ori >= 2 * math.pi:
                    ori -= 2 * math.pi

                o_bin = ori * n / (2.0 * math.pi)

                # Gaussian spatial weight
                g_weight = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_w2))
                val = mag * g_weight

                # Trilinear interpolation in x, y, orientation
                ix = math.floor(cx_bin)
                iy = math.floor(cy_bin)
                io = math.floor(o_bin)

                fx = cx_bin - ix
                fy = cy_bin - iy
                fo = o_bin - io

                for yy_i, wy in ((iy, 1 - fy), (iy + 1, fy)):
                    if yy_i < 0 or yy_i >= d:
                        continue
                    for xx_i, wx in ((ix, 1 - fx), (ix + 1, fx)):
                        if xx_i < 0 or xx_i >= d:
                            continue
                        for oo_i, wo in ((io % n, 1 - fo), ((io + 1) % n, fo)):
                            hist[yy_i, xx_i, oo_i] += val * wx * wy * wo

        desc = hist.reshape(-1)

        # Standard SIFT-style normalization
        norm = np.linalg.norm(desc)
        if norm < self.cfg.eps:
            return None
        desc = desc / norm
        desc = np.clip(desc, 0.0, clip_value)
        norm = np.linalg.norm(desc)
        if norm < self.cfg.eps:
            return None
        desc = desc / norm
        return desc.astype(np.float32)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    img1 = cv2.imread("input.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("distorted_image.jpg", cv2.IMREAD_COLOR)

    # xi is the radial distortion parameter.
    # You need to estimate it from calibration or roughly tune it.
    # Positive/negative sign depends on your distortion convention.
    cfg = SRDSIFTConfig(
        xi=0.15,
        center=None,  # defaults to image center
    )
    srd = SRDSIFT(cfg)

    kps1, des1 = srd.detect_and_compute(img1)
    kps2, des2 = srd.detect_and_compute(img2)
    matches = srd.match(des1, des2, ratio_test=0.75)

    vis = cv2.drawMatches(
        img1, kps1, img2, kps2, matches[:100], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite("srd_sift_matches.png", vis)

    print(f"Image 1: {len(kps1)} keypoints")
    print(f"Image 2: {len(kps2)} keypoints")
    print(f"Good matches: {len(matches)}")