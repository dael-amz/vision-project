import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


Array = np.ndarray


@dataclass
class DistortionModel:
    """
    Division-model radial distortion used by the sRD-SIFT paper.

    Coordinates are assumed to be normalized around `center` by `norm_scale`:
        x_n = (x - cx) / norm_scale
        y_n = (y - cy) / norm_scale

    The forward mapping used in the paper is x = u / (1 + xi * ||u||^2).
    This implementation works directly in distorted-image coordinates and uses
    the closed-form Jacobian reported in the paper for implicit gradient correction.

    IMPORTANT:
      * xi is NOT OpenCV's k1.
      * For barrel distortion in the paper's convention, xi is typically negative.
      * The center can be approximate; the paper reports reasonable robustness.
    """

    xi: float
    center: Tuple[float, float]
    norm_scale: float

    @classmethod
    def from_image_shape(
        cls,
        shape: Tuple[int, int],
        xi: float,
        center: Optional[Tuple[float, float]] = None,
        norm_scale: Optional[float] = None,
    ) -> "DistortionModel":
        h, w = shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        if norm_scale is None:
            norm_scale = max(h, w) * 0.5
        return cls(xi=xi, center=center, norm_scale=norm_scale)

    def normalize_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        cx, cy = self.center
        s = self.norm_scale
        return (x - cx) / s, (y - cy) / s

    def radius2_from_pixels(self, x: Array, y: Array) -> Array:
        xn, yn = self.normalize_points(x, y)
        return xn * xn + yn * yn

    def scale_factor(self, x: Array, y: Array) -> Array:
        """Adaptive Gaussian scale factor 1 + xi r^2 from Eq. (15)."""
        r2 = self.radius2_from_pixels(x, y)
        sf = 1.0 + self.xi * r2
        return np.maximum(sf, 1e-4)

    def jacobian_from_pixels(self, x: Array, y: Array) -> Array:
        """
        Jacobian J_f in distorted-image coordinates as reported in the paper.
        Returns array shape (..., 2, 2).
        """
        xn, yn = self.normalize_points(x, y)
        r2 = xn * xn + yn * yn
        xi = self.xi
        denom = np.maximum(1.0 - xi * r2, 1e-8)
        pref = (1.0 + xi * r2) / denom

        j11 = pref * (1.0 - xi * (r2 - 8.0 * xn * xn))
        j12 = pref * (8.0 * xi * xn * yn)
        j21 = j12
        j22 = pref * (1.0 - xi * (r2 - 8.0 * yn * yn))

        J = np.empty(xn.shape + (2, 2), dtype=np.float32)
        J[..., 0, 0] = j11
        J[..., 0, 1] = j12
        J[..., 1, 0] = j21
        J[..., 1, 1] = j22
        return J


def _to_gray_float32(image: Array) -> Array:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.5:
        image /= 255.0
    return image


def _gaussian_kernel_1d(sigma: float) -> Array:
    sigma = max(float(sigma), 1e-4)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


class _AdaptiveSeparableGaussian:
    """
    Separable adaptive Gaussian approximation used by sRD-SIFT.

    Uses per-column sigma on the horizontal pass and per-row sigma on the
    vertical pass, where sigma_eff = (1 + xi r^2) * sigma at the convolution
    center, following the paper's approximation in Eqs. (15)-(16).
    """

    def __init__(self, model: DistortionModel, lut_quantization: int = 256):
        self.model = model
        self.lut_quantization = lut_quantization
        self._cache = {}

    def _quantize_sigma(self, sigma: float) -> float:
        q = max(1e-4, float(sigma))
        return round(q * self.lut_quantization) / self.lut_quantization

    def _kernel(self, sigma: float) -> Array:
        key = self._quantize_sigma(sigma)
        if key not in self._cache:
            self._cache[key] = _gaussian_kernel_1d(key)
        return self._cache[key]

    def blur(self, image: Array, sigma: float) -> Array:
        h, w = image.shape
        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        cx, cy = self.model.center
        sf_x = self.model.scale_factor(xs, np.full_like(xs, cy))
        sf_y = self.model.scale_factor(np.full_like(ys, cx), ys)
        sigma_x = np.maximum(sf_x * sigma, 1e-4)
        sigma_y = np.maximum(sf_y * sigma, 1e-4)

        # Horizontal pass
        temp = np.empty_like(image, dtype=np.float32)
        for col in range(w):
            k = self._kernel(float(sigma_x[col]))
            rad = len(k) // 2
            left = max(0, col - rad)
            right = min(w, col + rad + 1)
            k_left = rad - (col - left)
            k_right = k_left + (right - left)
            weights = k[k_left:k_right]
            weights = weights / np.sum(weights)
            temp[:, col] = np.sum(image[:, left:right] * weights[None, :], axis=1)

        # Vertical pass
        out = np.empty_like(image, dtype=np.float32)
        for row in range(h):
            k = self._kernel(float(sigma_y[row]))
            rad = len(k) // 2
            top = max(0, row - rad)
            bottom = min(h, row + rad + 1)
            k_top = rad - (row - top)
            k_bottom = k_top + (bottom - top)
            weights = k[k_top:k_bottom]
            weights = weights / np.sum(weights)
            out[row, :] = np.sum(temp[top:bottom, :] * weights[:, None], axis=0)

        return out


@dataclass
class _OctaveData:
    gaussians: List[Array]
    dogs: List[Array]
    sigmas: List[float]
    scales_per_level: List[float]


class SRDSIFT:
    """
    Runnable sRD-SIFT implementation compatible with cv2.KeyPoint.

    This is designed to stay close to the paper:
      - adaptive scale-space via separable radius-aware Gaussian filtering
      - DoG keypoint detection on the distorted image plane
      - implicit gradient correction using the paper's Jacobian
      - SIFT-style orientation assignment and descriptor construction

    Notes:
      - xi follows the division-model paper, not OpenCV's k1.
      - The descriptor output is 128-D float32 like SIFT.
      - Keypoints are returned as cv2.KeyPoint objects.
    """

    def __init__(
        self,
        num_octaves: int = 4,
        num_intervals: int = 3,
        sigma: float = 1.6,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        orientation_bins: int = 36,
        descriptor_width: int = 4,
        descriptor_hist_bins: int = 8,
        descriptor_magnif: float = 3.0,
        max_interp_steps: int = 5,
        peak_ratio: float = 0.8,
    ) -> None:
        self.num_octaves = int(num_octaves)
        self.num_intervals = int(num_intervals)
        self.sigma = float(sigma)
        self.contrast_threshold = float(contrast_threshold)
        self.edge_threshold = float(edge_threshold)
        self.orientation_bins = int(orientation_bins)
        self.descriptor_width = int(descriptor_width)
        self.descriptor_hist_bins = int(descriptor_hist_bins)
        self.descriptor_magnif = float(descriptor_magnif)
        self.max_interp_steps = int(max_interp_steps)
        self.peak_ratio = float(peak_ratio)

        self._image: Optional[Array] = None
        self._model: Optional[DistortionModel] = None
        self._grad_x: Optional[Array] = None
        self._grad_y: Optional[Array] = None

    def detectAndCompute(
        self,
        image: Array,
        xi: float,
        center: Optional[Tuple[float, float]] = None,
        norm_scale: Optional[float] = None,
        mask: Optional[Array] = None,
    ) -> Tuple[List[cv2.KeyPoint], Array]:
        gray = _to_gray_float32(image)
        if mask is not None:
            gray = gray.copy()
            gray[mask == 0] = 0.0

        model = DistortionModel.from_image_shape(gray.shape, xi, center, norm_scale)
        self._image = gray
        self._model = model
        self._grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        self._grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)

        pyramid = self._build_pyramid(gray, model)
        keypoints = self._detect_keypoints(pyramid)
        keypoints = self._assign_orientations(keypoints)
        descriptors = self._compute_descriptors(keypoints)
        return keypoints, descriptors

    def compute(
        self,
        image: Array,
        keypoints: Sequence[cv2.KeyPoint],
        xi: float,
        center: Optional[Tuple[float, float]] = None,
        norm_scale: Optional[float] = None,
    ) -> Tuple[List[cv2.KeyPoint], Array]:
        gray = _to_gray_float32(image)
        model = DistortionModel.from_image_shape(gray.shape, xi, center, norm_scale)
        self._image = gray
        self._model = model
        self._grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        self._grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        desc = self._compute_descriptors(list(keypoints))
        return list(keypoints), desc

    def detect(
        self,
        image: Array,
        xi: float,
        center: Optional[Tuple[float, float]] = None,
        norm_scale: Optional[float] = None,
        mask: Optional[Array] = None,
    ) -> List[cv2.KeyPoint]:
        kps, _ = self.detectAndCompute(image, xi, center, norm_scale, mask)
        return kps

    def _build_pyramid(self, image: Array, model: DistortionModel) -> List[_OctaveData]:
        k = 2.0 ** (1.0 / self.num_intervals)
        num_levels = self.num_intervals + 3
        blur = _AdaptiveSeparableGaussian(model)

        current = image
        pyramid: List[_OctaveData] = []
        for octave in range(self.num_octaves):
            sigmas = [self.sigma * (k ** i) for i in range(num_levels)]
            gaussians = [blur.blur(current, s) for s in sigmas]
            dogs = [gaussians[i + 1] - gaussians[i] for i in range(len(gaussians) - 1)]
            scales_per_level = [sigmas[i] * (2 ** octave) for i in range(num_levels)]
            pyramid.append(_OctaveData(gaussians=gaussians, dogs=dogs, sigmas=sigmas, scales_per_level=scales_per_level))

            # Base of next octave = image downsampled by 2 from level num_intervals.
            base = gaussians[self.num_intervals]
            h, w = base.shape
            if min(h, w) < 32:
                break
            current = cv2.resize(base, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            cx, cy = model.center
            model = DistortionModel(
                xi=model.xi,
                center=(cx / 2.0, cy / 2.0),
                norm_scale=model.norm_scale / 2.0,
            )
            blur = _AdaptiveSeparableGaussian(model)
        return pyramid

    def _is_extremum(self, dogs: List[Array], layer: int, r: int, c: int) -> bool:
        val = dogs[layer][r, c]
        patch_prev = dogs[layer - 1][r - 1:r + 2, c - 1:c + 2]
        patch_cur = dogs[layer][r - 1:r + 2, c - 1:c + 2]
        patch_next = dogs[layer + 1][r - 1:r + 2, c - 1:c + 2]
        if val > 0:
            return val >= patch_prev.max() and val >= patch_cur.max() and val >= patch_next.max()
        return val <= patch_prev.min() and val <= patch_cur.min() and val <= patch_next.min()

    def _edge_response_ok(self, dog: Array, r: int, c: int) -> bool:
        val = dog[r, c]
        dxx = dog[r, c + 1] + dog[r, c - 1] - 2.0 * val
        dyy = dog[r + 1, c] + dog[r - 1, c] - 2.0 * val
        dxy = 0.25 * (dog[r + 1, c + 1] - dog[r + 1, c - 1] - dog[r - 1, c + 1] + dog[r - 1, c - 1])
        tr = dxx + dyy
        det = dxx * dyy - dxy * dxy
        if det <= 1e-12:
            return False
        rth = self.edge_threshold
        return (tr * tr) < ((rth + 1.0) * (rth + 1.0) / rth) * det

    def _detect_keypoints(self, pyramid: List[_OctaveData]) -> List[cv2.KeyPoint]:
        keypoints: List[cv2.KeyPoint] = []
        contrast_thr = 0.5 * self.contrast_threshold / self.num_intervals

        for octave_idx, octave in enumerate(pyramid):
            dogs = octave.dogs
            for layer in range(1, len(dogs) - 1):
                dog = dogs[layer]
                h, w = dog.shape
                for r in range(1, h - 1):
                    for c in range(1, w - 1):
                        val = dog[r, c]
                        if abs(float(val)) < contrast_thr:
                            continue
                        if not self._is_extremum(dogs, layer, r, c):
                            continue
                        if not self._edge_response_ok(dog, r, c):
                            continue

                        sigma_level = octave.sigmas[layer]
                        scale = 2 ** octave_idx
                        x = float(c * scale)
                        y = float(r * scale)
                        size = 2.0 * sigma_level * scale
                        response = float(abs(val))
                        kp = cv2.KeyPoint(x=x, y=y, size=size, angle=-1.0, response=response, octave=octave_idx, class_id=layer)
                        keypoints.append(kp)
        return keypoints

    def _corrected_gradients_at(self, xs: Array, ys: Array) -> Tuple[Array, Array]:
        assert self._image is not None and self._model is not None
        gx = cv2.remap(self._grad_x, xs.astype(np.float32), ys.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        gy = cv2.remap(self._grad_y, xs.astype(np.float32), ys.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        J = self._model.jacobian_from_pixels(xs, ys)
        gxu = J[..., 0, 0] * gx + J[..., 0, 1] * gy
        gyu = J[..., 1, 0] * gx + J[..., 1, 1] * gy
        return gxu, gyu

    def _assign_orientations(self, keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        oriented: List[cv2.KeyPoint] = []
        bins = self.orientation_bins
        for kp in keypoints:
            sigma = 1.5 * kp.size / 2.0
            radius = max(1, int(round(3.0 * sigma)))
            x0, y0 = kp.pt
            xs = np.arange(int(round(x0)) - radius, int(round(x0)) + radius + 1, dtype=np.float32)
            ys = np.arange(int(round(y0)) - radius, int(round(y0)) + radius + 1, dtype=np.float32)
            X, Y = np.meshgrid(xs, ys)

            if X.size == 0:
                continue
            gxu, gyu = self._corrected_gradients_at(X, Y)
            mag = np.sqrt(gxu * gxu + gyu * gyu)
            ori = (np.degrees(np.arctan2(gyu, gxu)) + 360.0) % 360.0

            if self._model is None:
                continue
            sigma_eff = sigma * self._model.scale_factor(X, Y)
            dx = X - x0
            dy = Y - y0
            weights = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_eff * sigma_eff))
            weighted = mag * weights

            hist = np.zeros(bins, dtype=np.float32)
            binf = ori * (bins / 360.0)
            bin0 = np.floor(binf).astype(np.int32) % bins
            frac = binf - np.floor(binf)
            bin1 = (bin0 + 1) % bins
            np.add.at(hist, bin0.ravel(), (weighted * (1.0 - frac)).ravel())
            np.add.at(hist, bin1.ravel(), (weighted * frac).ravel())

            hist = self._smooth_hist(hist)
            maxv = float(hist.max())
            if maxv <= 0:
                continue
            for i in range(bins):
                prev_i = (i - 1) % bins
                next_i = (i + 1) % bins
                if hist[i] > hist[prev_i] and hist[i] > hist[next_i] and hist[i] >= self.peak_ratio * maxv:
                    denom = hist[prev_i] - 2.0 * hist[i] + hist[next_i]
                    offset = 0.0 if abs(denom) < 1e-8 else 0.5 * (hist[prev_i] - hist[next_i]) / denom
                    angle = ((i + offset) * 360.0 / bins) % 360.0
                    new_kp = cv2.KeyPoint(x=kp.pt[0], y=kp.pt[1], size=kp.size, angle=float(angle), response=kp.response, octave=kp.octave, class_id=kp.class_id)
                    oriented.append(new_kp)
        return oriented

    @staticmethod
    def _smooth_hist(hist: Array) -> Array:
        out = hist.copy()
        for _ in range(2):
            out = (np.roll(out, -2) + 4*np.roll(out, -1) + 6*out + 4*np.roll(out, 1) + np.roll(out, 2)) / 16.0
        return out

    def _compute_descriptors(self, keypoints: List[cv2.KeyPoint]) -> Array:
        if len(keypoints) == 0:
            return np.zeros((0, self.descriptor_width * self.descriptor_width * self.descriptor_hist_bins), dtype=np.float32)
        descriptors = np.zeros((len(keypoints), self.descriptor_width * self.descriptor_width * self.descriptor_hist_bins), dtype=np.float32)
        for i, kp in enumerate(keypoints):
            descriptors[i, :] = self._descriptor_for_keypoint(kp)
        return descriptors

    def _descriptor_for_keypoint(self, kp: cv2.KeyPoint) -> Array:
        assert self._model is not None
        d = self.descriptor_width
        n = self.descriptor_hist_bins
        scale = kp.size / 2.0
        hist_width = self.descriptor_magnif * scale
        radius = int(round(hist_width * math.sqrt(2) * (d + 1) * 0.5))

        angle = math.radians(kp.angle)
        cos_t = math.cos(angle)
        sin_t = math.sin(angle)
        x0, y0 = kp.pt

        xs = np.arange(int(round(x0)) - radius, int(round(x0)) + radius + 1, dtype=np.float32)
        ys = np.arange(int(round(y0)) - radius, int(round(y0)) + radius + 1, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        dx = X - x0
        dy = Y - y0

        # Rotate into keypoint frame.
        x_rot = (cos_t * dx + sin_t * dy) / hist_width
        y_rot = (-sin_t * dx + cos_t * dy) / hist_width

        valid = (np.abs(x_rot) < (d / 2 + 0.5)) & (np.abs(y_rot) < (d / 2 + 0.5))
        if not np.any(valid):
            return np.zeros(d * d * n, dtype=np.float32)

        gxu, gyu = self._corrected_gradients_at(X, Y)
        mag = np.sqrt(gxu * gxu + gyu * gyu)
        ori = (np.degrees(np.arctan2(gyu, gxu)) - kp.angle + 360.0) % 360.0
        sigma_eff = hist_width * self._model.scale_factor(X, Y)
        weight = np.exp(-(x_rot * x_rot + y_rot * y_rot) / (2.0 * (0.5 * d) ** 2)) * mag
        # Paper note: replace Gaussian weighting with G(x,y;(1+xi r^2)sigma).
        # We approximate that by modulating the effective local sigma in the weight.
        weight *= np.clip(sigma_eff / max(hist_width, 1e-6), 1e-3, 1e3)

        hist = np.zeros((d, d, n), dtype=np.float32)

        xb = x_rot + d / 2 - 0.5
        yb = y_rot + d / 2 - 0.5
        ob = ori * (n / 360.0)

        x0b = np.floor(xb).astype(np.int32)
        y0b = np.floor(yb).astype(np.int32)
        o0b = np.floor(ob).astype(np.int32)
        dxw = xb - x0b
        dyw = yb - y0b
        dow = ob - o0b

        coords = np.argwhere(valid)
        for rr, cc in coords:
            xi0 = x0b[rr, cc]
            yi0 = y0b[rr, cc]
            oi0 = o0b[rr, cc]
            w = weight[rr, cc]
            if w <= 0:
                continue
            for yy, wy in ((yi0, 1 - dyw[rr, cc]), (yi0 + 1, dyw[rr, cc])):
                if yy < 0 or yy >= d:
                    continue
                for xx, wx in ((xi0, 1 - dxw[rr, cc]), (xi0 + 1, dxw[rr, cc])):
                    if xx < 0 or xx >= d:
                        continue
                    for oo, wo in ((oi0 % n, 1 - dow[rr, cc]), ((oi0 + 1) % n, dow[rr, cc])):
                        hist[yy, xx, oo] += w * wx * wy * wo

        desc = hist.ravel().astype(np.float32)
        norm = np.linalg.norm(desc)
        if norm > 1e-12:
            desc /= norm
        desc = np.clip(desc, 0, 0.2)
        norm = np.linalg.norm(desc)
        if norm > 1e-12:
            desc /= norm
        return desc


def match_descriptors(desc1: Array, desc2: Array, ratio: float = 0.75) -> List[cv2.DMatch]:
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(desc1, desc2, k=2)
    good: List[cv2.DMatch] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def draw_keypoints(image: Array, keypoints: Sequence[cv2.KeyPoint]) -> Array:
    return cv2.drawKeypoints(image, list(keypoints), None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def _demo() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run sRD-SIFT on an image.")
    parser.add_argument("image", help="Input image")
    parser.add_argument("--xi", type=float, required=True, help="Division-model distortion parameter xi")
    parser.add_argument("--center", type=float, nargs=2, default=None, help="Distortion center cx cy in pixels")
    parser.add_argument("--norm-scale", type=float, default=None, help="Normalization scale in pixels; default=max(H,W)/2")
    parser.add_argument("--out", default="srd_sift_keypoints.png", help="Output visualization path")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)

    detector = SRDSIFT()
    keypoints, desc = detector.detectAndCompute(image, xi=args.xi, center=None if args.center is None else tuple(args.center), norm_scale=args.norm_scale)
    vis = draw_keypoints(image, keypoints)
    cv2.imwrite(args.out, vis)
    print(f"Detected {len(keypoints)} keypoints")
    print(f"Descriptor shape: {desc.shape}")
    print(f"Saved visualization to: {args.out}")


if __name__ == "__main__":
    _demo()
