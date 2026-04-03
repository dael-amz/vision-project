from dataclasses import dataclass
from typing import List, Optional
from water_surface_simulator import WaterSurfaceSimulator
from skimage.feature import SIFT
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matching import Keypoints, D_MATCHER


import cv2
import numpy as np


@dataclass
class FBFlowTrackResult:
    initial_point: np.ndarray              # shape (2,)
    survived: bool
    num_successful_steps: int
    mean_fb_error: float
    max_fb_error: float
    fb_errors: List[float]
    positions: List[Optional[np.ndarray]]  # tracked position in each frame


class ForwardBackwardFlowFilter:
    """
    Track frame-0 keypoints across a grayscale frame sequence using
    pyramidal Lucas-Kanade optical flow.

    Quality metric:
        fb_error = || p_t - backward(forward(p_t)) ||

    Lower forward-backward error means more reliable tracking.
    """

    def __init__(
        self,
        win_size=(21, 21),
        max_level=3,
        max_iter=30,
        eps=1e-3,
        fb_error_threshold=1.5 * 20,
        min_successful_steps=1,
    ):
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                max_iter,
                eps,
            ),
        )
        self.fb_error_threshold = float(fb_error_threshold)
        self.min_successful_steps = int(min_successful_steps)

    def filter_points(self, frames: List[np.ndarray], points0: np.ndarray):
        """
        Parameters
        ----------
        frames : list of grayscale images
            frames[0], frames[1], ..., frames[T-1]
        points0 : ndarray of shape (N, 2)
            Keypoint coordinates in frame 0.

        Returns
        -------
        dict with:
            results         : list of FBFlowTrackResult
            survivor_mask   : bool array of shape (N,)
            survivor_points0: array of surviving frame-0 points
            mean_fb_errors  : array of shape (N,)
            tracks          : array of shape (N, num_frames, 2), NaN where unavailable
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames.")
        if points0.ndim != 2 or points0.shape[1] != 2:
            raise ValueError("points0 must have shape (N, 2).")

        num_frames = len(frames)
        num_points = points0.shape[0]

        points0 = points0.astype(np.float32)

        # tracks[i, t] = position of point i at frame t
        # NaN means tracking failed / unavailable
        tracks = np.full((num_points, num_frames, 2), np.nan, dtype=np.float32)
        tracks[:, 0, :] = points0

        results = []

        for i in range(num_points):
            p = points0[i].reshape(1, 1, 2)
            positions = [points0[i].copy()]
            fb_errors = []
            survived = True
            successful_steps = 0

            for t in range(num_frames - 1):
                img_t = frames[t]
                img_tp1 = frames[t + 1]

                # Forward: t -> t+1
                p_fwd, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
                    img_t, img_tp1, p, None, **self.lk_params
                )

                if p_fwd is None or st_fwd is None or st_fwd[0, 0] == 0:
                    survived = False
                    positions.append(None)
                    break

                # Backward: t+1 -> t
                p_bwd, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
                    img_tp1, img_t, p_fwd, None, **self.lk_params
                )

                if p_bwd is None or st_bwd is None or st_bwd[0, 0] == 0:
                    survived = False
                    positions.append(None)
                    break

                fb_error = float(np.linalg.norm(p.reshape(2) - p_bwd.reshape(2)))
                fb_errors.append(fb_error)

                new_pos = p_fwd.reshape(2).copy()
                tracks[i, t + 1, :] = new_pos
                positions.append(new_pos)

                if fb_error > self.fb_error_threshold:
                    survived = False
                    break

                successful_steps += 1
                p = p_fwd

            mean_fb_error = float(np.mean(fb_errors)) if fb_errors else np.inf
            max_fb_error = float(np.max(fb_errors)) if fb_errors else np.inf

            final_survived = survived and (successful_steps >= self.min_successful_steps)

            results.append(
                FBFlowTrackResult(
                    initial_point=points0[i].copy(),
                    survived=final_survived,
                    num_successful_steps=successful_steps,
                    mean_fb_error=mean_fb_error,
                    max_fb_error=max_fb_error,
                    fb_errors=fb_errors,
                    positions=positions,
                )
            )

        survivor_mask = np.array([r.survived for r in results], dtype=bool)
        mean_fb_errors = np.array([r.mean_fb_error for r in results], dtype=np.float32)
        survivor_points0 = points0[survivor_mask]

        return {
            "results": results,
            "survivor_mask": survivor_mask,
            "survivor_points0": survivor_points0,
            "mean_fb_errors": mean_fb_errors,
            "tracks": tracks,
        }
    

# # frames: list of grayscale images
# # keypoints0: ndarray of shape (N, 2), from your srd_sift detector on frame 0
# img = cv2.imread("input.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sim = WaterSurfaceSimulator(gray, amplitude_range=(0.00002, 0.00004), n_waves=10)
# frames = sim.generate_frames()

# desc = SIFT()
# desc.detect_and_extract(gray)

# # skimage SIFT keypoints are usually (row, col), so convert to (x, y)
# keypoints0 = desc.keypoints[:, ::-1].astype(np.float32)

# fb_filter = ForwardBackwardFlowFilter(
#     win_size=(21, 21),
#     max_level=3,
#     fb_error_threshold=1.5,
#     min_successful_steps=len(frames) - 1,
# )

# out = fb_filter.filter_points(frames, keypoints0)

# survivor_points0 = out["survivor_points0"]
# survivor_mask = out["survivor_mask"]
# tracks = out["tracks"]
# mean_fb_errors = out["mean_fb_errors"]

# print("tracks shape:", tracks.shape)
# print("frame 0 valid:", np.sum(~np.isnan(tracks[:, 0, 0])))
# if tracks.shape[1] > 1:
#     print("frame 1 valid:", np.sum(~np.isnan(tracks[:, 1, 0])))
#     print("mean displacement 0->1:",
#           np.nanmean(np.linalg.norm(tracks[:, 1, :] - tracks[:, 0, :], axis=1)))

# print(survivor_points0.shape)

# dist_func = sim.make_distortion_func(t=0.0)


# sif = SIFT()
# sif.detect_and_extract(gray)
# kp0 = sif.keypoints
# desc0 = sif.descriptors
# scales0 = sif.sigmas
# origin_kps = Keypoints(kp0[:, ::-1], desc0, scales0)
# print(len(kp0))

# kp1 = desc.keypoints[survivor_mask]
# desc1 = desc.descriptors[survivor_mask]
# scales1 = desc.sigmas[survivor_mask]
# distorted_kps = Keypoints(kp1[:, ::-1], desc1, scales1)
# print(len(kp1))

# matcher =  matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=distorted_kps, distortion_func=dist_func)
# out = matcher.compute_stats()
# print(out['repeatability'], out['recall'], out['precision'])

# kp2 = desc.keypoints
# desc2 = desc.descriptors
# scales2 = desc.sigmas
# distorted_kps = Keypoints(kp2[:, ::-1], desc2, scales2)
# matcher =  matcher = D_MATCHER(origin_kps=origin_kps, distorted_kps=distorted_kps, distortion_func=dist_func)
# out = matcher.compute_stats()
# print(out['repeatability'], out['recall'], out['precision'])

# fig, ax = plt.subplots()

# # show first image
# im = ax.imshow(frames[0], cmap="gray", animated=True)

# # initialize scatter with only valid points
# pts0 = tracks[:, 0, :]
# valid0 = ~np.isnan(pts0).any(axis=1)
# scat = ax.scatter(pts0[valid0, 0], pts0[valid0, 1], s=10, c="r",)# animated=True)

# ax.set_xlim(0, frames[0].shape[1])
# ax.set_ylim(frames[0].shape[0], 0)

# def init():
#     im.set_data(frames[0])
#     pts = tracks[:, 0, :]
#     valid = ~np.isnan(pts).any(axis=1)
#     scat.set_offsets(pts[valid])
#     return im, scat

# def update(i):
#     im.set_data(frames[i])
#     pts = tracks[:, i, :]
#     valid = ~np.isnan(pts).any(axis=1)
#     scat.set_offsets(pts[valid])
#     ax.set_title(f"frame {i}")
#     return im, scat

# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(frames),
#     init_func=init,
#     interval=200,
#     blit=False,   # important for debugging
#     repeat=True,
# )

# plt.show()

# print("initial keypoints:", len(keypoints0))
# print("surviving keypoints:", len(survivor_points0))