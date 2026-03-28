
import cv2
import numpy as np


class WaterSurfaceSimulator:
    """
    Simple dynamic water-surface distortion simulator.

    Geometry
    --------
    - camera center at z = 0
    - mean water surface at z = -h
    - underwater image plane at z = -(h + d)

    What this class does
    --------------------
    1. Render a distorted frame at time t.
    2. Generate a whole video (list of frames or MP4).
    3. Build a distortion function, similar in spirit to
       make_division_distortion_func(...), that maps keypoint locations/scales
       from the undistorted image to the distorted image at a chosen time.

    Water-surface intersection
    --------------------------
    This version computes the real intersection of each camera ray with the
    dynamic height field using Newton iterations. That makes the geometry more
    realistic than the mean-plane approximation, while still keeping the code
    reasonably compact.
    """

    def __init__(
        self,
        image,
        h=1.0,
        d=1.0,
        f=500.0,
        dt=0.05,
        duration=2.0,
        n_air=1.0,
        n_water=1.333,
        n_waves=12,
        wavelength_px_range=(30.0, 180.0),
        amplitude_range=(0.002, 0.02),
        seed=None,
    ):
        self.image = image
        self.h = float(h)
        self.d = float(d)
        self.f = float(f)
        self.dt = float(dt)
        self.duration = float(duration)
        self.n_air = float(n_air)
        self.n_water = float(n_water)

        self.rng = np.random.default_rng(seed)

        if image.ndim == 2:
            self.height_px, self.width_px = image.shape
        else:
            self.height_px, self.width_px = image.shape[:2]

        self.cx = (self.width_px - 1) * 0.5
        self.cy = (self.height_px - 1) * 0.5

        # Chosen so that flat-water center magnification is close to identity.
        self.eta = self.n_air / self.n_water
        self.plane_scale = (self.h + self.d * self.eta) / self.f

        self._prepare_camera_grid()

        self.waves = self._generate_random_waves(
            n_waves=n_waves,
            wavelength_px_range=wavelength_px_range,
            amplitude_range=amplitude_range,
        )

    # ------------------------------------------------------------------
    # Camera / wave setup
    # ------------------------------------------------------------------

    def _prepare_camera_grid(self):
        yy, xx = np.mgrid[0:self.height_px, 0:self.width_px].astype(np.float64)

        self.x = (xx - self.cx) / self.f
        self.y = (yy - self.cy) / self.f

        I = np.stack([self.x, self.y, -np.ones_like(self.x)], axis=-1)
        self.I = I / (np.linalg.norm(I, axis=-1, keepdims=True) + 1e-12)

        # Mean-plane hit points used by the small-slope approximation.
        self.X_mean = self.h * self.x
        self.Y_mean = self.h * self.y

    def _generate_random_waves(self, n_waves, wavelength_px_range, amplitude_range):
        """
        Random traveling sine waves.

        Wavelengths are sampled in *apparent pixel size near the image center*,
        then converted to world units using h / f.
        """
        waves = []

        world_per_pixel = self.h / self.f
        lam_min_px, lam_max_px = wavelength_px_range
        amp_min, amp_max = amplitude_range

        for _ in range(n_waves):
            lam_px = np.exp(self.rng.uniform(np.log(lam_min_px), np.log(lam_max_px)))
            lam_world = lam_px * world_per_pixel
            k_mag = 2.0 * np.pi / lam_world

            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            kx = k_mag * np.cos(theta)
            ky = k_mag * np.sin(theta)

            amp = self.rng.uniform(amp_min, amp_max)
            amp = min(amp, 0.25 / k_mag)  # keep slopes gentle

            omega = np.sqrt(9.81 * k_mag)  # deep-water-inspired
            phi = self.rng.uniform(0.0, 2.0 * np.pi)

            waves.append({
                "A": amp,
                "kx": kx,
                "ky": ky,
                "omega": omega,
                "phi": phi,
            })

        return waves

    # ------------------------------------------------------------------
    # Surface model
    # ------------------------------------------------------------------

    def _surface_height_and_gradient_at_points(self, X, Y, t):
        """
        Height and gradients at arbitrary world-plane points X, Y.

        X, Y can be scalars or arrays with the same shape.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        zeta = np.zeros_like(X)
        dzdx = np.zeros_like(X)
        dzdy = np.zeros_like(X)

        for w in self.waves:
            phase = w["kx"] * X + w["ky"] * Y - w["omega"] * t + w["phi"]
            s = np.sin(phase)
            c = np.cos(phase)

            zeta += w["A"] * s
            dzdx += w["A"] * w["kx"] * c
            dzdy += w["A"] * w["ky"] * c

        return zeta, dzdx, dzdy

    def surface_height_and_gradient(self, t):
        """
        Height and gradients on the full image-sized world grid corresponding
        to the mean-plane hit points.

        This is mostly useful for debugging / visualization.
        """
        return self._surface_height_and_gradient_at_points(self.X_mean, self.Y_mean, t)

    def _intersect_surface(self, I, t, max_iter=12, tol=1e-7):
        """
        Intersect rays with the true dynamic water surface.

        Parameters
        ----------
        I : array (..., 3)
            Unit ray directions from the camera center.
        t : float
            Time.
        max_iter : int
            Newton iterations.
        tol : float
            Convergence tolerance on the ray parameter.

        Returns
        -------
        P : array (..., 3)
            Intersection points on the water surface.
        zeta, dzdx, dzdy : arrays matching the leading shape of I
            Surface height and gradients at the intersection points.
        """
        # Ray: R(s) = s * I, since the camera center is at (0, 0, 0).
        # Surface: z = -h + zeta(x, y, t)
        # Solve F(s) = z_ray(s) - z_surface(x_ray(s), y_ray(s), t) = 0
        #
        # Good initial guess: intersection with the mean plane z = -h
        s = (-self.h) / (I[..., 2] + 1e-12)

        for _ in range(max_iter):
            X = s * I[..., 0]
            Y = s * I[..., 1]
            Z = s * I[..., 2]

            zeta, dzdx, dzdy = self._surface_height_and_gradient_at_points(X, Y, t)
            F = Z - (-self.h + zeta)

            # dF/ds = Iz - zeta_x * Ix - zeta_y * Iy
            dF = I[..., 2] - dzdx * I[..., 0] - dzdy * I[..., 1]

            delta = F / (dF + 1e-12)
            s_new = s - delta

            if np.max(np.abs(s_new - s)) < tol:
                s = s_new
                break
            s = s_new

        X = s * I[..., 0]
        Y = s * I[..., 1]
        Z = s * I[..., 2]
        zeta, dzdx, dzdy = self._surface_height_and_gradient_at_points(X, Y, t)

        P = np.stack([X, Y, Z], axis=-1)
        return P, zeta, dzdx, dzdy

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refract(self, I, N):
        """
        Vector Snell refraction from air to water.

        I : incident unit direction
        N : unit surface normal pointing toward air
        """
        eta = self.n_air / self.n_water
        cos_i = -np.sum(I * N, axis=-1, keepdims=True)
        k = 1.0 - eta**2 * (1.0 - cos_i**2)
        k = np.maximum(k, 0.0)
        T = eta * I + (eta * cos_i - np.sqrt(k)) * N
        T /= (np.linalg.norm(T, axis=-1, keepdims=True) + 1e-12)
        return T

    def _source_coords_from_distorted_pixels(self, uv, t):
        """
        For distorted-image pixel locations uv, return the corresponding source
        (undistorted image) pixel locations.

        This is the same mapping used internally by render_frame, but evaluated
        at arbitrary pixel positions instead of on the full image grid.

        Parameters
        ----------
        uv : array of shape (N, 2), in (x, y) = (col, row) pixel convention
        t : float

        Returns
        -------
        src_uv : array of shape (N, 2)
        """
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim == 1:
            uv = uv[None, :]

        u = uv[:, 0]
        v = uv[:, 1]

        x = (u - self.cx) / self.f
        y = (v - self.cy) / self.f

        I = np.stack([x, y, -np.ones_like(x)], axis=-1)
        I /= (np.linalg.norm(I, axis=-1, keepdims=True) + 1e-12)

        P, _, dzdx, dzdy = self._intersect_surface(I, t)

        N = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
        N /= (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12)

        T = self._refract(I, N)

        tau = (-(self.h + self.d) - P[:, 2]) / (T[:, 2] + 1e-12)
        Q = P + tau[:, None] * T

        src_u = Q[:, 0] / self.plane_scale + self.cx
        src_v = Q[:, 1] / self.plane_scale + self.cy

        return np.column_stack([src_u, src_v])

    def render_frame(self, t):
        """
        Render one distorted frame at time t.
        """
        P, _, dzdx, dzdy = self._intersect_surface(self.I, t)

        N = np.stack(
            [-dzdx, -dzdy, np.ones_like(dzdx)],
            axis=-1,
        )
        N /= (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12)

        T = self._refract(self.I, N)

        tau = (-(self.h + self.d) - P[..., 2]) / (T[..., 2] + 1e-12)
        Q = P + tau[..., None] * T

        map_x = Q[..., 0] / self.plane_scale + self.cx
        map_y = Q[..., 1] / self.plane_scale + self.cy

        frame = cv2.remap(
            self.image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return frame

    def generate_frames(self):
        times = np.arange(0.0, self.duration, self.dt)
        return [self.render_frame(t) for t in times]

    def save_video(self, path, fps=None):
        if fps is None:
            fps = 1.0 / self.dt

        frames = self.generate_frames()
        if len(frames) == 0:
            raise ValueError("No frames were generated.")

        h, w = frames[0].shape[:2]
        is_color = (frames[0].ndim == 3)

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
            isColor=is_color,
        )

        for frame in frames:
            out = frame
            if frame.ndim == 2:
                out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(out)

        writer.release()

    # ------------------------------------------------------------------
    # Keypoint distortion function
    # ------------------------------------------------------------------

    def distort_keypoints(
        self,
        t,
        origin_loc,
        origin_scales,
        max_iter=8,
        tol=1e-4,
        eps=1.0,
        scale_mode="area",
    ):
        """
        Map undistorted keypoints to distorted keypoints at time t.

        Parameters
        ----------
        t : float
            Time of the distorted frame.
        origin_loc : array, shape (N, 2)
            Keypoint locations in the undistorted image, in (x, y) pixel convention.
        origin_scales : array, shape (N,)
            Keypoint scales in the undistorted image.
        max_iter : int
            Newton iterations used to invert the pixel warp.
        tol : float
            Stop if pixel residual gets below this.
        eps : float
            Finite-difference step used for the local Jacobian, in pixels.
        scale_mode : {"none", "area"}
            - "none": keep the input scales unchanged
            - "area": scale by inverse sqrt(det(J)), where J is the local
              Jacobian of source_coords(distorted_pixel). This is a simple
              isotropic approximation to local magnification.

        Returns
        -------
        distorted_loc : array, shape (N, 2)
        distorted_scales : array, shape (N,)
        """
        origin_loc = np.asarray(origin_loc, dtype=np.float64)
        origin_scales = np.asarray(origin_scales, dtype=np.float64)

        if origin_loc.ndim != 2 or origin_loc.shape[1] != 2:
            raise ValueError("origin_loc must have shape (N, 2)")
        if origin_scales.ndim != 1 or origin_scales.shape[0] != origin_loc.shape[0]:
            raise ValueError("origin_scales must have shape (N,)")

        n = origin_loc.shape[0]
        distorted_loc = np.zeros_like(origin_loc)
        distorted_scales = np.zeros_like(origin_scales)

        # Start from identity. Good enough when the distortion is not extreme.
        guess = origin_loc.copy()

        for i in range(n):
            p = guess[i].copy()
            target_src = origin_loc[i]

            for _ in range(max_iter):
                src = self._source_coords_from_distorted_pixels(p[None, :], t)[0]
                r = src - target_src

                if np.linalg.norm(r) < tol:
                    break

                # Finite-difference Jacobian: d(source_uv) / d(distorted_uv)
                src_dx = self._source_coords_from_distorted_pixels(
                    np.array([[p[0] + eps, p[1]]]), t
                )[0]
                src_dy = self._source_coords_from_distorted_pixels(
                    np.array([[p[0], p[1] + eps]]), t
                )[0]

                J = np.column_stack([
                    (src_dx - src) / eps,
                    (src_dy - src) / eps,
                ])

                # Newton step for F(p) = source_coords(p) - target_src
                try:
                    delta = np.linalg.solve(J, r)
                except np.linalg.LinAlgError:
                    break

                p = p - delta

            distorted_loc[i] = p

            if scale_mode == "none":
                distorted_scales[i] = origin_scales[i]
            elif scale_mode == "area":
                src = self._source_coords_from_distorted_pixels(p[None, :], t)[0]
                src_dx = self._source_coords_from_distorted_pixels(
                    np.array([[p[0] + eps, p[1]]]), t
                )[0]
                src_dy = self._source_coords_from_distorted_pixels(
                    np.array([[p[0], p[1] + eps]]), t
                )[0]

                J = np.column_stack([
                    (src_dx - src) / eps,
                    (src_dy - src) / eps,
                ])

                detJ = np.linalg.det(J)
                local_scale = 1.0 / np.sqrt(max(abs(detJ), 1e-12))
                distorted_scales[i] = origin_scales[i] * local_scale
            else:
                raise ValueError(f"Unknown scale_mode: {scale_mode}")

        return distorted_loc, distorted_scales

    def make_distortion_func(self, t, scale_mode="area", max_iter=8, tol=1e-4, eps=1.0):
        """
        Return a function with the same interface style as make_division_distortion_func.

        Example
        -------
        dist_func = sim.make_distortion_func(t=0.5)
        distorted_loc, distorted_scales = dist_func(origin_loc, origin_scales)
        """
        def distortion_func(origin_loc, origin_scales):
            return self.distort_keypoints(
                t=t,
                origin_loc=origin_loc,
                origin_scales=origin_scales,
                max_iter=max_iter,
                tol=tol,
                eps=eps,
                scale_mode=scale_mode,
            )
        return distortion_func


if __name__ == "__main__":
    img = cv2.imread("/mnt/data/example_input.png")
    if img is None:
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        for y in range(0, 400, 40):
            cv2.line(img, (0, y), (599, y), (255, 255, 255), 1)
        for x in range(0, 600, 40):
            cv2.line(img, (x, 0), (x, 399), (255, 255, 255), 1)
        cv2.putText(
            img,
            "Water Surface Simulator",
            (60, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    sim = WaterSurfaceSimulator(
        image=img,
        h=1.0,
        d=1.0,
        f=500.0,
        dt=0.04,
        duration=3.0,
        n_waves=16,
        wavelength_px_range=(25, 140),
        amplitude_range=(0.002, 0.015),
        seed=0,
    )

    sim.save_video("/mnt/data/water_surface_simulation.mp4")

    # Small example for the keypoint distortion function.
    keypoints = np.array([[100.0, 100.0], [300.0, 200.0], [500.0, 300.0]])
    scales = np.array([3.0, 5.0, 7.0])

    dist_func = sim.make_distortion_func(t=0.5)
    distorted_kp, distorted_scales = dist_func(keypoints, scales)

    print("Undistorted keypoints:")
    print(keypoints)
    print("Distorted keypoints:")
    print(distorted_kp)
    print("Distorted scales:")
    print(distorted_scales)
