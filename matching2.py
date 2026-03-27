from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np
from skimage.feature import match_descriptors


Array = np.ndarray


@dataclass
class Keypoints:
    """
    Minimal keypoint container.

    xy:    (N, 2) array of [x, y] positions
    scale: (N,)   array of keypoint scales
    desc:  (N, D) descriptors
    """
    xy: Array
    scale: Array
    desc: Optional[Array] = None


def default_scale_consistency(
    mapped_scale: Array,
    ref_scale: Array,
    log2_tol: float = 0.5,
) -> Array:
    """
    True when scales differ by at most log2_tol octaves.
    """
    eps = 1e-12
    return np.abs(np.log2((mapped_scale + eps) / (ref_scale + eps))) <= log2_tol


def greedy_unique_assignment(
    query_idx: Array,
    train_idx: Array,
    score: Array,
) -> Tuple[Array, Array, Array]:
    """
    Enforce one-to-one correspondences greedily by lowest score first.
    Mostly useful for detector repeatability pairing.
    """
    order = np.argsort(score)
    used_q = set()
    used_t = set()
    keep = []

    for k in order:
        q = int(query_idx[k])
        t = int(train_idx[k])
        if q in used_q or t in used_t:
            continue
        used_q.add(q)
        used_t.add(t)
        keep.append(k)

    keep = np.asarray(keep, dtype=int)
    return query_idx[keep], train_idx[keep], score[keep]


def compute_common_detections(
    kp_distorted: Keypoints,
    kp_original: Keypoints,
    map_xy_and_scale_to_original: Callable[[Array, Array], Tuple[Array, Array]],
    pos_tol: float = 3.0,
    scale_log2_tol: float = 0.5,
    enforce_one_to_one: bool = True,
) -> Dict[str, Any]:
    """
    Approximate S_true using:
      - mapped position consistency
      - mapped scale consistency
    """
    xy_d = np.asarray(kp_distorted.xy, dtype=float)
    sc_d = np.asarray(kp_distorted.scale, dtype=float)
    xy_o = np.asarray(kp_original.xy, dtype=float)
    sc_o = np.asarray(kp_original.scale, dtype=float)

    mapped_xy, mapped_sc = map_xy_and_scale_to_original(xy_d, sc_d)

    disp = mapped_xy[:, None, :] - xy_o[None, :, :]
    pos_dist = np.linalg.norm(disp, axis=2)

    sc_cons = default_scale_consistency(
        mapped_sc[:, None],
        sc_o[None, :],
        log2_tol=scale_log2_tol,
    )

    valid = (pos_dist <= pos_tol) & sc_cons
    rows, cols = np.nonzero(valid)

    if len(rows) == 0:
        denom = min(len(xy_d), len(xy_o))
        return {
            "S_true_count": 0,
            "denominator": denom,
            "repeatability": 0.0,
            "distorted_indices": np.array([], dtype=int),
            "original_indices": np.array([], dtype=int),
        }

    q_idx = rows.astype(int)
    t_idx = cols.astype(int)
    score = pos_dist[rows, cols]

    if enforce_one_to_one:
        q_idx, t_idx, score = greedy_unique_assignment(q_idx, t_idx, score)

    denom = min(len(xy_d), len(xy_o))
    repeatability = len(q_idx) / denom if denom > 0 else 0.0

    return {
        "S_true_count": len(q_idx),
        "denominator": denom,
        "repeatability": repeatability,
        "distorted_indices": q_idx,
        "original_indices": t_idx,
    }


def skimage_match(
    desc_query: Array,
    desc_train: Array,
    *,
    metric: Optional[str] = "euclidean",
    p: int = 2,
    max_distance: np.floating | float | None = np.inf,
    cross_check: bool = True,
    max_ratio: float = 0.8,
) -> Tuple[Array, Array]:
    """
    Wrapper around skimage.feature.match_descriptors.

    Returns:
        query_idx, train_idx
    """
    desc_query = np.asarray(desc_query)
    desc_train = np.asarray(desc_train)

    if desc_query.ndim != 2 or desc_train.ndim != 2:
        raise ValueError("Descriptors must be 2D arrays of shape (N, D).")
    if desc_query.shape[1] != desc_train.shape[1]:
        raise ValueError("Descriptor dimensionalities do not match.")

    matches = match_descriptors(
        desc_query,
        desc_train,
        metric=metric,
        p=p,
        max_distance=max_distance,
        cross_check=cross_check,
        max_ratio=max_ratio,
    )

    if matches.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    return matches[:, 0].astype(int), matches[:, 1].astype(int)


def evaluate_matching(
    kp_distorted: Keypoints,
    kp_original: Keypoints,
    map_xy_and_scale_to_original: Callable[[Array, Array], Tuple[Array, Array]],
    *,
    pos_tol: float = 3.0,
    scale_log2_tol: float = 0.5,
    metric: Optional[str] = "euclidean",
    p: int = 2,
    max_distance: np.floating | float | None = np.inf,
    cross_check: bool = True,
    max_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    Compute:
      - repeatability
      - recall
      - precision

    using skimage.feature.match_descriptors for descriptor matching.
    """
    if kp_distorted.desc is None or kp_original.desc is None:
        raise ValueError("Descriptors are required for recall/precision.")

    common = compute_common_detections(
        kp_distorted=kp_distorted,
        kp_original=kp_original,
        map_xy_and_scale_to_original=map_xy_and_scale_to_original,
        pos_tol=pos_tol,
        scale_log2_tol=scale_log2_tol,
        enforce_one_to_one=True,
    )
    S_true_count = common["S_true_count"]

    q_idx, t_idx = skimage_match(
        kp_distorted.desc,
        kp_original.desc,
        metric=metric,
        p=p,
        max_distance=max_distance,
        cross_check=cross_check,
        max_ratio=max_ratio,
    )

    M_count = len(q_idx)

    if M_count == 0:
        return {
            "repeatability": common["repeatability"],
            "recall": 0.0,
            "precision": 0.0,
            "S_true_count": S_true_count,
            "M_count": 0,
            "M_true_count": 0,
            "matches_query_idx": q_idx,
            "matches_train_idx": t_idx,
            "matches_correct_mask": np.array([], dtype=bool),
        }

    mapped_xy, mapped_sc = map_xy_and_scale_to_original(
        kp_distorted.xy[q_idx],
        kp_distorted.scale[q_idx],
    )

    pos_err = np.linalg.norm(mapped_xy - kp_original.xy[t_idx], axis=1)
    sc_ok = default_scale_consistency(
        mapped_sc,
        kp_original.scale[t_idx],
        log2_tol=scale_log2_tol,
    )

    correct = (pos_err <= pos_tol) & sc_ok
    M_true_count = int(np.sum(correct))

    recall = M_true_count / S_true_count if S_true_count > 0 else 0.0
    precision = M_true_count / M_count if M_count > 0 else 0.0

    return {
        "repeatability": common["repeatability"],
        "recall": recall,
        "precision": precision,
        "S_true_count": S_true_count,
        "M_count": M_count,
        "M_true_count": M_true_count,
        "matches_query_idx": q_idx,
        "matches_train_idx": t_idx,
        "matches_correct_mask": correct,
    }