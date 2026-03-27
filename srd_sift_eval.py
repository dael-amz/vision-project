
"""
Synthetic precision/recall evaluation for sRD-SIFT using the original slower srd_sift.py.

This script:
1. Loads a single original image
2. Generates a synthetic radially distorted version itself
3. Runs:
      - baseline SIFT on the original image
      - baseline SIFT on the distorted image
      - original sRD-SIFT on the distorted image
4. Matches distorted-image descriptors back to SIFT descriptors from the original image
5. Computes precision and recall using the known synthetic distortion map
6. Saves visualizations:
      - distorted image
      - baseline SIFT matches
      - sRD-SIFT matches
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from srd_sift import SRDSIFT


Array = np.ndarray


@dataclass
class EvalResult:
    name: str
    n_keypoints_original: int
    n_keypoints_distorted: int
    n_matches: int
    n_true_positive_matches: int
    precision: float
    recall: float
    recoverable_original_keypoints: int
    original_keypoints_total: int


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_gray_u8(image: Array) -> Array:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        if image.max() <= 1.5:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _kp_points(keypoints: Sequence[cv2.KeyPoint]) -> Array:
    if len(keypoints) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)


def _distort_points_division(
    pts: Array,
    xi: float,
    center: Tuple[float, float],
    norm_scale: float,
) -> Array:
    pts = np.asarray(pts, dtype=np.float32)
    xn = (pts[:, 0] - center[0]) / norm_scale
    yn = (pts[:, 1] - center[1]) / norm_scale
    r2 = xn * xn + yn * yn
    scale = 1.0 / np.maximum(1.0 + xi * r2, 1e-8)
    xd = xn * scale
    yd = yn * scale
    out = np.empty_like(pts)
    out[:, 0] = xd * norm_scale + center[0]
    out[:, 1] = yd * norm_scale + center[1]
    return out


def generate_distorted_image(
    image: Array,
    xi: float,
    center: Optional[Tuple[float, float]] = None,
    norm_scale: Optional[float] = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[Array, Tuple[float, float], float]:
    h, w = image.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    if norm_scale is None:
        norm_scale = max(h, w) * 0.5

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xd = (xx - center[0]) / norm_scale
    yd = (yy - center[1]) / norm_scale
    rd2 = xd * xd + yd * yd

    inv_scale = 1.0 / np.maximum(1.0 - xi * rd2, 1e-8)
    xu = xd * inv_scale
    yu = yd * inv_scale

    map_x = xu * norm_scale + center[0]
    map_y = yu * norm_scale + center[1]

    distorted = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return distorted, center, float(norm_scale)


def match_descriptors_ratio(
    des1: Optional[Array],
    des2: Optional[Array],
    ratio: float = 0.75,
) -> List[cv2.DMatch]:
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


def evaluate_matches(
    name: str,
    kp_original: Sequence[cv2.KeyPoint],
    kp_distorted: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    gt_distorted_points_for_original: Array,
    tolerance_px: float = 5.0,
) -> Tuple[EvalResult, List[cv2.DMatch], List[cv2.DMatch]]:
    pts_distorted_detected = _kp_points(kp_distorted)
    total_orig = len(kp_original)

    recoverable = 0
    if len(pts_distorted_detected) > 0 and len(gt_distorted_points_for_original) > 0:
        diff = gt_distorted_points_for_original[:, None, :] - pts_distorted_detected[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        recoverable_mask = np.any(d2 <= tolerance_px * tolerance_px, axis=1)
        recoverable = int(np.sum(recoverable_mask))
    else:
        recoverable_mask = np.zeros((total_orig,), dtype=bool)

    good_matches: List[cv2.DMatch] = []
    bad_matches: List[cv2.DMatch] = []
    tp_original_indices = set()

    for m in matches:
        qi = m.queryIdx
        ti = m.trainIdx
        if qi >= len(kp_original) or ti >= len(kp_distorted):
            bad_matches.append(m)
            continue

        gt_pt = gt_distorted_points_for_original[qi]
        det_pt = np.array(kp_distorted[ti].pt, dtype=np.float32)
        dist = float(np.linalg.norm(det_pt - gt_pt))
        if dist <= tolerance_px:
            good_matches.append(m)
            tp_original_indices.add(qi)
        else:
            bad_matches.append(m)

    precision = float(len(good_matches) / max(len(matches), 1))
    recall = float(len(tp_original_indices) / max(recoverable, 1))

    result = EvalResult(
        name=name,
        n_keypoints_original=len(kp_original),
        n_keypoints_distorted=len(kp_distorted),
        n_matches=len(matches),
        n_true_positive_matches=len(good_matches),
        precision=precision,
        recall=recall,
        recoverable_original_keypoints=recoverable,
        original_keypoints_total=total_orig,
    )
    return result, good_matches, bad_matches


def draw_matches_split(
    img_original: Array,
    kp_original: Sequence[cv2.KeyPoint],
    img_distorted: Array,
    kp_distorted: Sequence[cv2.KeyPoint],
    good_matches: Sequence[cv2.DMatch],
    bad_matches: Sequence[cv2.DMatch],
    max_draw: int = 80,
) -> Array:
    good_draw = list(good_matches[:max_draw // 2])
    bad_draw = list(bad_matches[:max_draw // 2])

    left = cv2.drawMatches(
        img_original, list(kp_original),
        img_distorted, list(kp_distorted),
        good_draw, None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    right = cv2.drawMatches(
        img_original, list(kp_original),
        img_distorted, list(kp_distorted),
        bad_draw, None,
        matchColor=(0, 0, 255),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if left.shape[0] != right.shape[0]:
        h = max(left.shape[0], right.shape[0])

        def pad(img: Array) -> Array:
            if img.shape[0] == h:
                return img
            pad_amt = h - img.shape[0]
            return cv2.copyMakeBorder(img, 0, pad_amt, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        left = pad(left)
        right = pad(right)

    header_h = 40
    canvas = np.zeros((left.shape[0] + header_h, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[header_h:, :left.shape[1]] = left
    canvas[header_h:, left.shape[1]:] = right

    cv2.putText(canvas, "Correct matches (green)", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Incorrect matches (red)", (left.shape[1] + 20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


def print_result(result: EvalResult) -> None:
    print(f"\n[{result.name}]")
    print(f"Original keypoints (SIFT):     {result.n_keypoints_original}")
    print(f"Distorted keypoints:           {result.n_keypoints_distorted}")
    print(f"Total matches:                 {result.n_matches}")
    print(f"True-positive matches:         {result.n_true_positive_matches}")
    print(f"Recoverable original points:   {result.recoverable_original_keypoints}/{result.original_keypoints_total}")
    print(f"Precision:                     {result.precision:.4f}")
    print(f"Recall:                        {result.recall:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic precision/recall test for original sRD-SIFT")
    parser.add_argument("image", help="Path to the original input image")
    parser.add_argument("--xi", type=float, required=True, help="Synthetic distortion parameter (division-style)")
    parser.add_argument("--cx", type=float, default=None, help="Distortion center x")
    parser.add_argument("--cy", type=float, default=None, help="Distortion center y")
    parser.add_argument("--norm-scale", type=float, default=None, help="Normalization scale")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    parser.add_argument("--tolerance", type=float, default=5.0, help="Ground-truth pixel tolerance for correctness")
    parser.add_argument("--outdir", type=str, default="srd_eval_output", help="Directory for outputs")
    parser.add_argument("--draw", type=int, default=80, help="Max total matches to draw per method")
    parser.add_argument("--show", action="store_true", help="Show output visualizations in windows")
    args = parser.parse_args()

    _ensure_dir(args.outdir)

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    h, w = image.shape[:2]
    center = (
        (w - 1) * 0.5 if args.cx is None else args.cx,
        (h - 1) * 0.5 if args.cy is None else args.cy,
    )
    norm_scale = max(h, w) * 0.5 if args.norm_scale is None else float(args.norm_scale)

    distorted, center, norm_scale = generate_distorted_image(
        image=image,
        xi=args.xi,
        center=center,
        norm_scale=norm_scale,
    )
    cv2.imwrite(os.path.join(args.outdir, "distorted_image.png"), distorted)

    sift = cv2.SIFT_create()
    gray_original = _to_gray_u8(image)
    gray_distorted = _to_gray_u8(distorted)

    kp_orig, des_orig = sift.detectAndCompute(gray_original, None)
    kp_dist_sift, des_dist_sift = sift.detectAndCompute(gray_distorted, None)

    srd = SRDSIFT()
    kp_dist_srd, des_dist_srd = srd.detectAndCompute(
        distorted,
        xi=args.xi,
        center=center,
        norm_scale=norm_scale,
)

    gt_distorted_for_original = _distort_points_division(
        _kp_points(kp_orig),
        xi=args.xi,
        center=center,
        norm_scale=norm_scale,
    )

    matches_sift = match_descriptors_ratio(des_orig, des_dist_sift, ratio=args.ratio)
    matches_srd = match_descriptors_ratio(des_orig, des_dist_srd, ratio=args.ratio)

    res_sift, good_sift, bad_sift = evaluate_matches(
        name="Baseline SIFT(original) -> SIFT(distorted)",
        kp_original=kp_orig,
        kp_distorted=kp_dist_sift,
        matches=matches_sift,
        gt_distorted_points_for_original=gt_distorted_for_original,
        tolerance_px=args.tolerance,
    )

    res_srd, good_srd, bad_srd = evaluate_matches(
        name="SIFT(original) -> original sRD-SIFT(distorted)",
        kp_original=kp_orig,
        kp_distorted=kp_dist_srd,
        matches=matches_srd,
        gt_distorted_points_for_original=gt_distorted_for_original,
        tolerance_px=args.tolerance,
    )

    print_result(res_sift)
    print_result(res_srd)

    vis_sift = draw_matches_split(
        image, kp_orig, distorted, kp_dist_sift, good_sift, bad_sift, max_draw=args.draw
    )
    vis_srd = draw_matches_split(
        image, kp_orig, distorted, kp_dist_srd, good_srd, bad_srd, max_draw=args.draw
    )

    cv2.imwrite(os.path.join(args.outdir, "matches_baseline_sift.png"), vis_sift)
    cv2.imwrite(os.path.join(args.outdir, "matches_srd_sift.png"), vis_srd)

    dist_sift_kp = cv2.drawKeypoints(
        distorted, kp_dist_sift, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    dist_srd_kp = cv2.drawKeypoints(
        distorted, kp_dist_srd, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(os.path.join(args.outdir, "distorted_keypoints_sift.png"), dist_sift_kp)
    cv2.imwrite(os.path.join(args.outdir, "distorted_keypoints_srd.png"), dist_srd_kp)

    summary_path = os.path.join(args.outdir, "metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for result in (res_sift, res_srd):
            f.write(f"[{result.name}]\n")
            f.write(f"Original keypoints (SIFT): {result.n_keypoints_original}\n")
            f.write(f"Distorted keypoints: {result.n_keypoints_distorted}\n")
            f.write(f"Total matches: {result.n_matches}\n")
            f.write(f"True-positive matches: {result.n_true_positive_matches}\n")
            f.write(f"Recoverable original points: {result.recoverable_original_keypoints}/{result.original_keypoints_total}\n")
            f.write(f"Precision: {result.precision:.6f}\n")
            f.write(f"Recall: {result.recall:.6f}\n\n")

    print(f"\nSaved outputs to: {args.outdir}")
    print("- distorted_image.png")
    print("- matches_baseline_sift.png")
    print("- matches_srd_sift.png")
    print("- distorted_keypoints_sift.png")
    print("- distorted_keypoints_srd.png")
    print("- metrics.txt")

    if args.show:
        cv2.imshow("Distorted image", distorted)
        cv2.imshow("Baseline SIFT matches", vis_sift)
        cv2.imshow("Original sRD-SIFT matches", vis_srd)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
