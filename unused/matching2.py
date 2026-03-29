import numpy as np
import skimage.feature as features


class Keypoints:
    locations: np.ndarray
    descriptors: np.ndarray
    scales: np.ndarray

    def __init__(self, locations, descriptors, scales):
        self.locations = np.asarray(locations, dtype=np.float64)
        self.descriptors = np.asarray(descriptors)
        self.scales = np.asarray(scales, dtype=np.float64)

    def loc(self):
        return self.locations

    def desc(self):
        return self.descriptors

    def sc(self):
        return self.scales

    def keypoint(self, index):
        kp = {
            "location": self.locations[index],
            "descriptors": self.descriptors[index],
            "scale": self.scales[index],
        }
        return kp


def circle_iou_matrix(C1, R1, C2, R2, eps=1e-9):
    """
    Pairwise IoU between two sets of circles.

    Parameters
    ----------
    C1 : (N, 2) array
    R1 : (N,) array
    C2 : (M, 2) array
    R2 : (M,) array

    Returns
    -------
    iou : (N, M) array
    """
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    R1 = np.asarray(R1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)

    diff = C1[:, None, :] - C2[None, :, :]
    d = np.linalg.norm(diff, axis=2)   # (N, M)

    r1 = R1[:, None]                   # (N, 1)
    r2 = R2[None, :]                   # (1, M)

    # Broadcast to full shape so boolean indexing works
    r1_full = np.broadcast_to(r1, d.shape)
    r2_full = np.broadcast_to(r2, d.shape)

    iou = np.zeros_like(d, dtype=np.float64)

    # Case 1: no overlap
    no_overlap = d >= (r1_full + r2_full)

    # Case 2: one circle fully inside the other
    inside = d <= np.abs(r1_full - r2_full)
    if np.any(inside):
        r_min = np.minimum(r1_full, r2_full)
        r_max = np.maximum(r1_full, r2_full)
        inter = np.pi * r_min**2
        union = np.pi * r_max**2
        iou[inside] = (inter / (union + eps))[inside]

    # Case 3: partial overlap
    partial = ~(no_overlap | inside)
    if np.any(partial):
        d_p = d[partial]
        r1_p = r1_full[partial]
        r2_p = r2_full[partial]

        arg1 = (d_p**2 + r1_p**2 - r2_p**2) / (2.0 * d_p * r1_p + eps)
        arg2 = (d_p**2 + r2_p**2 - r1_p**2) / (2.0 * d_p * r2_p + eps)

        arg1 = np.clip(arg1, -1.0, 1.0)
        arg2 = np.clip(arg2, -1.0, 1.0)

        alpha = np.arccos(arg1)
        beta = np.arccos(arg2)

        term = (
            (-d_p + r1_p + r2_p)
            * (d_p + r1_p - r2_p)
            * (d_p - r1_p + r2_p)
            * (d_p + r1_p + r2_p)
        )
        term = np.maximum(term, 0.0)

        inter = r1_p**2 * alpha + r2_p**2 * beta - 0.5 * np.sqrt(term)
        union = np.pi * r1_p**2 + np.pi * r2_p**2 - inter
        iou[partial] = inter / (union + eps)

    return iou


class D_MATCHER:
    distorted_kps: Keypoints
    origin_kps: Keypoints
    distortion_func: callable
    matches: np.ndarray
    s_true: np.ndarray
    M_true: np.ndarray

    def __init__(
        self,
        distorted_kps: Keypoints,
        origin_kps: Keypoints,
        distortion_func=lambda x, s: (x, s),
        overlap_thresh: float = 0.7,
        region_radius_factor: float = 1.0,
    ):
        """
        Parameters
        ----------
        distorted_kps, origin_kps : Keypoints
            Keypoints in distorted and undistorted images
        distortion_func : callable
            Maps undistorted keypoints into the distorted image frame:
                distorted_loc, distorted_scales = distortion_func(origin_loc, origin_scales)
        overlap_thresh : float
            Minimum IoU to count as a repeatable detection. The paper uses > 70%.
        region_radius_factor : float
            Converts scale/sigma into region radius:
                radius = region_radius_factor * scale
            Keep this the same for both images.
        """
        self.distorted_kps = distorted_kps
        self.origin_kps = origin_kps
        self.distortion_func = distortion_func
        self.overlap_thresh = float(overlap_thresh)
        self.region_radius_factor = float(region_radius_factor)

        self.matches = None
        self.s_true = None
        self.M_true = None

    def match_kp(self, max_ratio=0.8, cross_check=True):
        origin_desc = self.origin_kps.desc()
        distorted_desc = self.distorted_kps.desc()

        self.matches = features.match_descriptors(
            origin_desc,
            distorted_desc,
            cross_check=cross_check,
            max_ratio=max_ratio,
        )
        return self.matches

    @staticmethod
    def unique_assignment(query_idx, train_idx, score, maximize=False):
        """
        Enforce one-to-one correspondences greedily.

        If maximize=False, lower score is better.
        If maximize=True, higher score is better.
        """
        order = np.argsort(score)
        if maximize:
            order = order[::-1]

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

    def _pairwise_overlap_candidates(self, one_to_one=True):
        """
        Compare:
          predicted distorted regions from origin_kps
        against:
          detected distorted regions from distorted_kps

        Returns
        -------
        pairs : (K, 2) int array
            [origin_index, distorted_index]
        scores : (K,) float array
            IoU values for kept pairs
        """
        origin_loc = self.origin_kps.loc()
        origin_scales = self.origin_kps.sc()

        distorted_loc = self.distorted_kps.loc()
        distorted_scales = self.distorted_kps.sc()

        pred_loc, pred_scales = self.distortion_func(origin_loc, origin_scales)

        pred_r = self.region_radius_factor * np.asarray(pred_scales, dtype=np.float64)
        dist_r = self.region_radius_factor * np.asarray(distorted_scales, dtype=np.float64)

        iou = circle_iou_matrix(pred_loc, pred_r, distorted_loc, dist_r)

        rows, cols = np.where(iou >= self.overlap_thresh)
        if len(rows) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=np.float64)

        scores = iou[rows, cols]

        if one_to_one:
            rows, cols, scores = self.unique_assignment(
                rows, cols, scores, maximize=True
            )

        pairs = np.column_stack((rows, cols))
        return pairs, scores

    def compute_s_true(self, one_to_one=True):
        """
        Compute S_true using paper-style region overlap instead of point distance.
        """
        self.s_true, _ = self._pairwise_overlap_candidates(one_to_one=one_to_one)
        return self.s_true

    def true_matches(self, one_to_one=True):
        """
        Compute M_true: descriptor matches that are also geometrically true
        under the overlap criterion.
        """
        if self.matches is None:
            raise RuntimeError("Call match_kp before true_matches.")

        origin_idx = self.matches[:, 0]
        distorted_idx = self.matches[:, 1]

        origin_loc = self.origin_kps.loc()[origin_idx]
        origin_scales = self.origin_kps.sc()[origin_idx]

        distorted_loc = self.distorted_kps.loc()[distorted_idx]
        distorted_scales = self.distorted_kps.sc()[distorted_idx]

        pred_loc, pred_scales = self.distortion_func(origin_loc, origin_scales)

        pred_r = self.region_radius_factor * np.asarray(pred_scales, dtype=np.float64)
        dist_r = self.region_radius_factor * np.asarray(distorted_scales, dtype=np.float64)

        # IoU only for already-matched pairs
        iou = np.diag(circle_iou_matrix(pred_loc, pred_r, distorted_loc, dist_r))

        keep = iou >= self.overlap_thresh
        true_match_indices = np.where(keep)[0]

        if len(true_match_indices) == 0:
            self.M_true = np.empty((0, 2), dtype=int)
            return self.M_true

        if one_to_one:
            # Usually match_descriptors already gives one-to-one with cross_check=True,
            # but we keep this for safety.
            rows = origin_idx[true_match_indices]
            cols = distorted_idx[true_match_indices]
            scores = iou[true_match_indices]

            rows, cols, scores = self.unique_assignment(
                rows, cols, scores, maximize=True
            )
            self.M_true = np.column_stack((rows, cols))
        else:
            self.M_true = self.matches[keep]

        return self.M_true

    def compute_stats(self, max_ratio=0.8, cross_check=True, one_to_one=True):
        self.match_kp(max_ratio=max_ratio, cross_check=cross_check)
        self.compute_s_true(one_to_one=one_to_one)
        self.true_matches(one_to_one=one_to_one)

        n_dist = len(self.distorted_kps.loc())
        n_orig = len(self.origin_kps.loc())
        n_min = min(n_dist, n_orig)
        n_matches = len(self.matches)
        n_s_true = len(self.s_true)
        n_m_true = len(self.M_true)

        result = {
            "repeatability": (n_s_true / n_min) if n_min > 0 else 0.0,
            "recall": (n_m_true / n_s_true) if n_s_true > 0 else 0.0,
            "precision": (n_m_true / n_matches) if n_matches > 0 else 0.0,
        }

        return result