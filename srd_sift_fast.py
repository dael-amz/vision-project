
"""
Fast-ish practical sRD-SIFT implementation.

What this file provides
-----------------------
- SRDSIFT class with:
    - detect(image, xi, center=None, norm_scale=None)
    - compute(image, keypoints, xi, center=None, norm_scale=None)
    - detectAndCompute(image, xi, center=None, norm_scale=None)
- cv2.KeyPoint-compatible output
- 128-D float32 descriptors
- test / demo CLI with keypoint visualization

Main speed improvements over the previous version
-------------------------------------------------
1. Quantized adaptive blur: per-column/per-row sigma bins instead of per-pixel kernels
2. Incremental pyramid building: each level is blurred from the previous level
3. Vectorized DoG extrema detection using dilation/erosion
4. Cached corrected gradient maps per octave/layer
5. Descriptor extraction uses precomputed magnitude/angle maps
6. Optional limit on returned keypoints

Notes
-----
This stays aligned with the paper's main ideas:
- adaptive distorted-plane scale space
- DoG keypoint detection
- Jacobian-based implicit gradient correction
- SIFT-style orientation and descriptor construction

But, like any independent implementation, some engineering details are practical choices.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Array = np.ndarray


def _to_gray_float32(image: Array) -> Array:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.5:
        image /= 255.0
    return image


@dataclass
class DistortionModel:
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
        return cls(float(xi), center, float(norm_scale))

    def radius2_map(self, shape: Tuple[int, int]) -> Array:
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xn = (xx - self.center[0]) / self.norm_scale
        yn = (yy - self.center[1]) / self.norm_scale
        return xn * xn + yn * yn

    def resized(self, scale: float) -> "DistortionModel":
        cx, cy = self.center
        return DistortionModel(
            xi=self.xi,
            center=(cx / scale, cy / scale),
            norm_scale=self.norm_scale / scale,
        )


@dataclass
class PyramidLevel:
    image: Array
    sigma_abs: float
    grad_mag: Optional[Array] = None
    grad_ang: Optional[Array] = None


@dataclass
class OctaveData:
    gaussian_levels: List[PyramidLevel]
    dog_levels: List[Array]
    scale_factor: float
    model: DistortionModel


class AdaptiveBlurFast:
    """
    Fast adaptive separable blur.

    Instead of a unique sigma for every sample position, this groups columns and rows
    into a fixed number of quantized sigma bins and applies one 1-D convolution per bin.
    This keeps the spirit of the paper's efficient separable approximation while making
    Python execution much faster.
    """

    def __init__(self, sigma_bins: int = 24):
        self.sigma_bins = int(max(4, sigma_bins))
        self.kernel_cache: Dict[float, Array] = {}

    def _gaussian_kernel(self, sigma: float) -> Array:
        sigma = max(float(sigma), 1e-4)
        sigma_q = round(sigma, 3)
        if sigma_q in self.kernel_cache:
            return self.kernel_cache[sigma_q]
        radius = max(1, int(round(3.0 * sigma_q)))
        xs = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-(xs * xs) / (2.0 * sigma_q * sigma_q))
        k /= np.sum(k)
        self.kernel_cache[sigma_q] = k.astype(np.float32)
        return self.kernel_cache[sigma_q]

    def _bin_sigmas(self, sigmas: Array) -> Tuple[Array, Array]:
        smin = float(sigmas.min())
        smax = float(sigmas.max())
        if abs(smax - smin) < 1e-6:
            centers = np.array([smin], dtype=np.float32)
            labels = np.zeros(sigmas.shape, dtype=np.int32)
            return labels, centers
        centers = np.linspace(smin, smax, self.sigma_bins, dtype=np.float32)
        labels = np.abs(sigmas[:, None] - centers[None, :]).argmin(axis=1).astype(np.int32)
        return labels, centers

    def blur(self, image: Array, sigma_eff_map: Array) -> Array:
        h, w = image.shape

        col_sigmas = sigma_eff_map[h // 2, :].astype(np.float32)
        row_sigmas = sigma_eff_map[:, w // 2].astype(np.float32)

        col_labels, col_centers = self._bin_sigmas(col_sigmas)
        row_labels, row_centers = self._bin_sigmas(row_sigmas)

        temp = np.empty_like(image, dtype=np.float32)
        for bi, sigma in enumerate(col_centers):
            cols = np.where(col_labels == bi)[0]
            if cols.size == 0:
                continue
            kernel = self._gaussian_kernel(float(sigma))
            blurred = cv2.sepFilter2D(image, cv2.CV_32F, kernel, np.array([1.0], dtype=np.float32))
            temp[:, cols] = blurred[:, cols]

        out = np.empty_like(image, dtype=np.float32)
        for bi, sigma in enumerate(row_centers):
            rows = np.where(row_labels == bi)[0]
            if rows.size == 0:
                continue
            kernel = self._gaussian_kernel(float(sigma))
            blurred = cv2.sepFilter2D(temp, cv2.CV_32F, np.array([1.0], dtype=np.float32), kernel)
            out[rows, :] = blurred[rows, :]

        return out


class SRDSIFT:
    def __init__(
        self,
        num_octaves: int = 4,
        num_intervals: int = 3,
        sigma0: float = 1.6,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        orientation_bins: int = 36,
        descriptor_width: int = 4,
        descriptor_hist_bins: int = 8,
        descriptor_magnif: float = 3.0,
        peak_ratio: float = 0.8,
        sigma_bins: int = 24,
        max_keypoints: Optional[int] = 4000,
    ) -> None:
        self.num_octaves = int(num_octaves)
        self.num_intervals = int(num_intervals)
        self.sigma0 = float(sigma0)
        self.contrast_threshold = float(contrast_threshold)
        self.edge_threshold = float(edge_threshold)
        self.orientation_bins = int(orientation_bins)
        self.descriptor_width = int(descriptor_width)
        self.descriptor_hist_bins = int(descriptor_hist_bins)
        self.descriptor_magnif = float(descriptor_magnif)
        self.peak_ratio = float(peak_ratio)
        self.max_keypoints = max_keypoints

        self.blur_engine = AdaptiveBlurFast(sigma_bins=sigma_bins)

        self._gray: Optional[Array] = None
        self._base_model: Optional[DistortionModel] = None
        self._pyramid: Optional[List[OctaveData]] = None

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

    def compute(
        self,
        image: Array,
        keypoints: Sequence[cv2.KeyPoint],
        xi: float,
        center: Optional[Tuple[float, float]] = None,
        norm_scale: Optional[float] = None,
    ) -> Tuple[List[cv2.KeyPoint], Optional[Array]]:
        self._prepare(image, xi, center, norm_scale, None)
        desc = self._compute_descriptors(list(keypoints))
        return list(keypoints), desc

    def detectAndCompute(
        self,
        image: Array,
        xi: float,
        center: Optional[Tuple[float, float]] = None,
        norm_scale: Optional[float] = None,
        mask: Optional[Array] = None,
    ) -> Tuple[List[cv2.KeyPoint], Optional[Array]]:
        self._prepare(image, xi, center, norm_scale, mask)
        keypoints = self._detect_keypoints()
        keypoints = self._assign_orientations(keypoints)
        if self.max_keypoints is not None and len(keypoints) > self.max_keypoints:
            keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[: self.max_keypoints]
        desc = self._compute_descriptors(keypoints)
        return keypoints, desc

    def _prepare(
        self,
        image: Array,
        xi: float,
        center: Optional[Tuple[float, float]],
        norm_scale: Optional[float],
        mask: Optional[Array],
    ) -> None:
        gray = _to_gray_float32(image)
        if mask is not None:
            gray = gray.copy()
            gray[mask == 0] = 0.0

        self._gray = gray
        self._base_model = DistortionModel.from_image_shape(gray.shape, xi, center, norm_scale)
        self._pyramid = self._build_pyramid(gray, self._base_model)

    def _adaptive_sigma_map(self, shape: Tuple[int, int], sigma_inc: float, model: DistortionModel) -> Array:
        r2 = model.radius2_map(shape)
        sigma_map = (1.0 + model.xi * r2) * sigma_inc
        return np.maximum(sigma_map.astype(np.float32), 0.05)

    def _build_pyramid(self, base: Array, base_model: DistortionModel) -> List[OctaveData]:
        k = 2.0 ** (1.0 / self.num_intervals)
        pyramid: List[OctaveData] = []

        current_base = base
        current_model = base_model
        current_scale = 1.0

        for octave in range(self.num_octaves):
            levels: List[PyramidLevel] = []
            dogs: List[Array] = []

            abs_sigmas = [self.sigma0]
            for i in range(1, self.num_intervals + 3):
                abs_sigmas.append(self.sigma0 * (k ** i))

            levels.append(PyramidLevel(image=current_base, sigma_abs=abs_sigmas[0]))

            for i in range(1, len(abs_sigmas)):
                prev_sigma = abs_sigmas[i - 1]
                cur_sigma = abs_sigmas[i]
                sigma_inc = math.sqrt(max(cur_sigma * cur_sigma - prev_sigma * prev_sigma, 1e-8))
                sigma_map = self._adaptive_sigma_map(current_base.shape, sigma_inc, current_model)
                next_img = self.blur_engine.blur(levels[-1].image, sigma_map)
                levels.append(PyramidLevel(image=next_img, sigma_abs=cur_sigma))
                dogs.append(next_img - levels[-2].image)

            octave_data = OctaveData(
                gaussian_levels=levels,
                dog_levels=dogs,
                scale_factor=current_scale,
                model=current_model,
            )
            self._precompute_corrected_gradients(octave_data)
            pyramid.append(octave_data)

            if octave < self.num_octaves - 1:
                next_base_img = levels[self.num_intervals].image
                h, w = next_base_img.shape
                current_base = cv2.resize(next_base_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
                current_scale *= 2.0
                current_model = base_model.resized(current_scale)

        return pyramid

    def _precompute_corrected_gradients(self, octave_data: OctaveData) -> None:
        model = octave_data.model
        for level in octave_data.gaussian_levels:
            img = level.image
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

            h, w = img.shape
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            xn = (xx - model.center[0]) / model.norm_scale
            yn = (yy - model.center[1]) / model.norm_scale
            r2 = xn * xn + yn * yn

            denom = np.maximum(1.0 - model.xi * r2, 1e-8)
            pref = (1.0 + model.xi * r2) / denom
            j11 = pref * (1.0 - model.xi * (r2 - 8.0 * xn * xn))
            j12 = pref * (8.0 * model.xi * xn * yn)
            j22 = pref * (1.0 - model.xi * (r2 - 8.0 * yn * yn))

            gux = j11 * gx + j12 * gy
            guy = j12 * gx + j22 * gy

            level.grad_mag = np.sqrt(gux * gux + guy * guy).astype(np.float32)
            level.grad_ang = np.mod(np.arctan2(guy, gux), 2.0 * np.pi).astype(np.float32)

    def _dog_local_extrema_mask(self, prev_d: Array, cur_d: Array, next_d: Array) -> Array:
        kernel = np.ones((3, 3), dtype=np.uint8)

        cur_max = cv2.dilate(cur_d, kernel)
        prev_max = cv2.dilate(prev_d, kernel)
        next_max = cv2.dilate(next_d, kernel)

        cur_min = cv2.erode(cur_d, kernel)
        prev_min = cv2.erode(prev_d, kernel)
        next_min = cv2.erode(next_d, kernel)

        is_max = (cur_d >= cur_max) & (cur_d >= prev_max) & (cur_d >= next_max)
        is_min = (cur_d <= cur_min) & (cur_d <= prev_min) & (cur_d <= next_min)

        mask = (is_max | is_min) & (np.abs(cur_d) >= self.contrast_threshold)

        mask[[0, -1], :] = False
        mask[:, [0, -1]] = False
        return mask

    def _detect_keypoints(self) -> List[cv2.KeyPoint]:
        assert self._pyramid is not None
        keypoints: List[cv2.KeyPoint] = []
        k = 2.0 ** (1.0 / self.num_intervals)
        edge_limit = ((self.edge_threshold + 1.0) ** 2) / self.edge_threshold

        for octave_idx, octave in enumerate(self._pyramid):
            for layer in range(1, len(octave.dog_levels) - 1):
                prev_d = octave.dog_levels[layer - 1]
                cur_d = octave.dog_levels[layer]
                next_d = octave.dog_levels[layer + 1]

                mask = self._dog_local_extrema_mask(prev_d, cur_d, next_d)
                ys, xs = np.where(mask)
                if xs.size == 0:
                    continue

                dxx = cur_d[ys, xs + 1] + cur_d[ys, xs - 1] - 2.0 * cur_d[ys, xs]
                dyy = cur_d[ys + 1, xs] + cur_d[ys - 1, xs] - 2.0 * cur_d[ys, xs]
                dxy = (cur_d[ys + 1, xs + 1] - cur_d[ys + 1, xs - 1] - cur_d[ys - 1, xs + 1] + cur_d[ys - 1, xs - 1]) * 0.25

                det = dxx * dyy - dxy * dxy
                tr = dxx + dyy
                good = det > 1e-10
                good &= ((tr * tr) / np.maximum(det, 1e-10)) < edge_limit

                xs = xs[good]
                ys = ys[good]
                if xs.size == 0:
                    continue

                sigma = self.sigma0 * (k ** layer) * octave.scale_factor
                size = 2.0 * sigma

                responses = np.abs(cur_d[ys, xs])
                for x, y, response in zip(xs, ys, responses):
                    kp = cv2.KeyPoint(float(x * octave.scale_factor), float(y * octave.scale_factor), float(size))
                    kp.response = float(response)
                    kp.octave = int(octave_idx + (layer << 8))
                    kp.angle = -1.0
                    keypoints.append(kp)

        return keypoints

    @staticmethod
    def _unpack_octave_layer(kp: cv2.KeyPoint) -> Tuple[int, int]:
        octave = kp.octave & 255
        layer = (kp.octave >> 8) & 255
        return octave, layer

    def _orientation_hist(self, mag_patch: Array, ang_patch: Array, sigma: float) -> Array:
        h, w = mag_patch.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        dx = xx - cx
        dy = yy - cy
        weights = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).astype(np.float32)

        hist = np.zeros(self.orientation_bins, dtype=np.float32)
        bins = np.mod(ang_patch * (self.orientation_bins / (2.0 * np.pi)), self.orientation_bins)
        b0 = np.floor(bins).astype(np.int32)
        frac = bins - b0
        vals = (mag_patch * weights).astype(np.float32)

        np.add.at(hist, b0.ravel(), (vals * (1.0 - frac)).ravel())
        np.add.at(hist, ((b0 + 1) % self.orientation_bins).ravel(), (vals * frac).ravel())
        return hist

    def _assign_orientations(self, keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        assert self._pyramid is not None
        oriented: List[cv2.KeyPoint] = []

        for kp in keypoints:
            octave_idx, layer = self._unpack_octave_layer(kp)
            octave = self._pyramid[octave_idx]
            level = octave.gaussian_levels[layer]
            mag = level.grad_mag
            ang = level.grad_ang
            assert mag is not None and ang is not None

            scale = octave.scale_factor
            x0 = kp.pt[0] / scale
            y0 = kp.pt[1] / scale

            sigma = 1.5 * kp.size / (2.0 * scale)
            radius = max(1, int(round(3.0 * sigma)))

            x0i = int(round(x0))
            y0i = int(round(y0))
            x1 = max(1, x0i - radius)
            x2 = min(mag.shape[1] - 1, x0i + radius + 1)
            y1 = max(1, y0i - radius)
            y2 = min(mag.shape[0] - 1, y0i + radius + 1)

            if x2 <= x1 or y2 <= y1:
                continue

            hist = self._orientation_hist(mag[y1:y2, x1:x2], ang[y1:y2, x1:x2], sigma=max(sigma, 1e-3))
            peak = float(hist.max())
            if peak <= 1e-8:
                continue

            for bi, value in enumerate(hist):
                if value >= self.peak_ratio * peak:
                    angle_deg = (bi + 0.5) * (360.0 / self.orientation_bins)
                    out = cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, angle_deg, kp.response, kp.octave, -1)
                    oriented.append(out)

        return oriented

    def _compute_descriptors(self, keypoints: Sequence[cv2.KeyPoint]) -> Optional[Array]:
        assert self._pyramid is not None
        if len(keypoints) == 0:
            return None

        d = self.descriptor_width
        n = self.descriptor_hist_bins
        descriptors: List[Array] = []

        for kp in keypoints:
            octave_idx, layer = self._unpack_octave_layer(kp)
            octave = self._pyramid[octave_idx]
            level = octave.gaussian_levels[layer]
            mag = level.grad_mag
            ang = level.grad_ang
            assert mag is not None and ang is not None

            scale_img = octave.scale_factor
            x0 = kp.pt[0] / scale_img
            y0 = kp.pt[1] / scale_img

            theta0 = math.radians(kp.angle)
            cos_t = math.cos(theta0)
            sin_t = math.sin(theta0)

            hist = np.zeros((d, d, n), dtype=np.float32)

            scale = self.descriptor_magnif * kp.size / (2.0 * scale_img)
            radius = int(round(scale * math.sqrt(2.0) * (d + 1) * 0.5))
            if radius < 1:
                descriptors.append(np.zeros((d * d * n,), dtype=np.float32))
                continue

            x0i = int(round(x0))
            y0i = int(round(y0))
            x1 = max(1, x0i - radius)
            x2 = min(mag.shape[1] - 1, x0i + radius + 1)
            y1 = max(1, y0i - radius)
            y2 = min(mag.shape[0] - 1, y0i + radius + 1)
            if x2 <= x1 or y2 <= y1:
                descriptors.append(np.zeros((d * d * n,), dtype=np.float32))
                continue

            yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
            dx = xx - x0
            dy = yy - y0

            rx = (cos_t * dx + sin_t * dy) / max(scale, 1e-8)
            ry = (-sin_t * dx + cos_t * dy) / max(scale, 1e-8)

            xb = rx + d / 2.0 - 0.5
            yb = ry + d / 2.0 - 0.5

            valid = (xb > -1.0) & (xb < d) & (yb > -1.0) & (yb < d)
            if not np.any(valid):
                descriptors.append(np.zeros((d * d * n,), dtype=np.float32))
                continue

            local_mag = mag[y1:y2, x1:x2][valid]
            local_ang = ang[y1:y2, x1:x2][valid]
            local_dx = dx[valid]
            local_dy = dy[valid]
            local_xb = xb[valid]
            local_yb = yb[valid]

            xn = (x0 - octave.model.center[0]) / octave.model.norm_scale
            yn = (y0 - octave.model.center[1]) / octave.model.norm_scale
            r2 = xn * xn + yn * yn
            win_sigma = max((1.0 + octave.model.xi * r2) * scale, 1e-3)

            weight = np.exp(-(local_dx * local_dx + local_dy * local_dy) / (2.0 * win_sigma * win_sigma)).astype(np.float32)
            local_mag = local_mag * weight
            ob = np.mod((local_ang - theta0) * (n / (2.0 * np.pi)), n)

            xi0 = np.floor(local_xb).astype(np.int32)
            yi0 = np.floor(local_yb).astype(np.int32)
            oi0 = np.floor(ob).astype(np.int32)

            dxb = local_xb - xi0
            dyb = local_yb - yi0
            dob = ob - oi0

            for sy in (0, 1):
                yidx = yi0 + sy
                wy = (1.0 - dyb) if sy == 0 else dyb
                valid_y = (yidx >= 0) & (yidx < d)
                if not np.any(valid_y):
                    continue

                for sx in (0, 1):
                    xidx = xi0 + sx
                    wx = (1.0 - dxb) if sx == 0 else dxb
                    valid_xy = valid_y & (xidx >= 0) & (xidx < d)
                    if not np.any(valid_xy):
                        continue

                    for so in (0, 1):
                        oidx = (oi0 + so) % n
                        wo = (1.0 - dob) if so == 0 else dob
                        vv = valid_xy
                        contrib = local_mag[vv] * wx[vv] * wy[vv] * wo[vv]
                        np.add.at(hist, (yidx[vv], xidx[vv], oidx[vv]), contrib)

            vec = hist.ravel()
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                vec /= norm
                vec = np.clip(vec, 0.0, 0.2)
                norm2 = float(np.linalg.norm(vec))
                if norm2 > 1e-12:
                    vec /= norm2
            descriptors.append((vec * 512.0).clip(0, 255).astype(np.float32))

        return np.vstack(descriptors).astype(np.float32)

    @staticmethod
    def draw_keypoints(image: Array, keypoints: Sequence[cv2.KeyPoint]) -> Array:
        return cv2.drawKeypoints(
            image,
            list(keypoints),
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )


def match_descriptors(des1: Optional[Array], des2: Optional[Array], ratio: float = 0.75):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Fast-ish practical sRD-SIFT demo")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--xi", type=float, required=True, help="Division-model xi")
    parser.add_argument("--cx", type=float, default=None, help="Distortion center x")
    parser.add_argument("--cy", type=float, default=None, help="Distortion center y")
    parser.add_argument("--norm-scale", type=float, default=None, help="Normalization scale")
    parser.add_argument("--out", type=str, default="srd_sift_keypoints.png", help="Output visualization path")
    parser.add_argument("--show", action="store_true", help="Show visualization window")
    parser.add_argument("--max-keypoints", type=int, default=2000, help="Maximum keypoints to keep")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    h, w = image.shape[:2]
    center = (
        (w - 1) * 0.5 if args.cx is None else args.cx,
        (h - 1) * 0.5 if args.cy is None else args.cy,
    )

    detector = SRDSIFT(max_keypoints=args.max_keypoints)
    keypoints, descriptors = detector.detectAndCompute(
        image,
        xi=args.xi,
        center=center,
        norm_scale=args.norm_scale,
    )

    print(f"Detected keypoints: {len(keypoints)}")
    print(f"Descriptor shape: {None if descriptors is None else descriptors.shape}")

    vis = detector.draw_keypoints(image, keypoints)
    cv2.imwrite(args.out, vis)
    print(f"Saved visualization to: {args.out}")

    if args.show:
        cv2.imshow("sRD-SIFT keypoints", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _cli()
