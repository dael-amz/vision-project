import sys
import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ============================================================
# Basic helpers
# ============================================================

def normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def nothing(_):
    pass


def get_slider_float(window: str, name: str, scale: float = 100.0) -> float:
    return cv2.getTrackbarPos(name, window) / scale


def overlay_text(img: np.ndarray, lines: List[str], x: int = 10, y: int = 25):
    out = img.copy()
    for i, line in enumerate(lines):
        yy = y + i * 24
        cv2.putText(out, line, (x, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (x, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ============================================================
# Refraction
# ============================================================

def refract(incident: np.ndarray,
            normal: np.ndarray,
            n1: float,
            n2: float) -> np.ndarray:
    """
    Refract rays using Snell's law.
    incident: (...,3) unit vectors toward interface in medium n1
    normal:   (...,3) normals pointing from medium n2 -> n1
    """
    incident = normalize(incident)
    normal = normalize(normal)

    eta = n1 / n2
    cosi = -np.sum(normal * incident, axis=-1, keepdims=True)
    cosi = np.clip(cosi, -1.0, 1.0)

    k = 1.0 - eta**2 * (1.0 - cosi**2)
    valid = k >= 0.0

    sqrt_k = np.sqrt(np.maximum(k, 0.0))
    transmitted = eta * incident + (eta * cosi - sqrt_k) * normal
    transmitted = normalize(transmitted)

    invalid_mask = ~valid[..., 0]
    if np.any(invalid_mask):
        transmitted[invalid_mask] = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    return transmitted


# ============================================================
# Water surface
# ============================================================

@dataclass
class HeightField:
    h_map: np.ndarray
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    z0: float = 0.0

    def __post_init__(self):
        self.h_map = self.h_map.astype(np.float64)
        self.H, self.W = self.h_map.shape
        self.xmin, self.xmax = self.xlim
        self.ymin, self.ymax = self.ylim

        if self.W < 2 or self.H < 2:
            raise ValueError("HeightField grid must be at least 2x2.")

        self.dx = (self.xmax - self.xmin) / (self.W - 1)
        self.dy = (self.ymax - self.ymin) / (self.H - 1)
        self.dh_dy, self.dh_dx = np.gradient(self.h_map, self.dy, self.dx)

    def world_to_grid(self, x: np.ndarray, y: np.ndarray):
        gx = (x - self.xmin) / (self.xmax - self.xmin) * (self.W - 1)
        gy = (y - self.ymin) / (self.ymax - self.ymin) * (self.H - 1)
        gx = np.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
        gy = np.nan_to_num(gy, nan=0.0, posinf=0.0, neginf=0.0)
        return gx, gy

    def bilinear_sample(self, arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gx, gy = self.world_to_grid(x, y)

        x0 = np.floor(gx).astype(np.int32)
        y0 = np.floor(gy).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        x0 = np.clip(x0, 0, self.W - 1)
        x1 = np.clip(x1, 0, self.W - 1)
        y0 = np.clip(y0, 0, self.H - 1)
        y1 = np.clip(y1, 0, self.H - 1)

        wx = gx - x0
        wy = gy - y0

        a = arr[y0, x0]
        b = arr[y0, x1]
        c = arr[y1, x0]
        d = arr[y1, x1]

        return ((1 - wx) * (1 - wy) * a +
                wx * (1 - wy) * b +
                (1 - wx) * wy * c +
                wx * wy * d)

    def sample_height_and_normal(self, x: np.ndarray, y: np.ndarray):
        h = self.bilinear_sample(self.h_map, x, y)
        hx = self.bilinear_sample(self.dh_dx, x, y)
        hy = self.bilinear_sample(self.dh_dy, x, y)

        n = np.stack([-hx, -hy, np.ones_like(h)], axis=-1)
        n = normalize(n)

        z = self.z0 + h
        return z, n, hx, hy


@dataclass
class WaveComponent:
    amplitude: float
    wavelength: float
    angle: float
    phase: float
    speed: float


def random_wave_pool(num_waves: int,
                     rng: np.random.Generator,
                     amplitude_range: Tuple[float, float] = (0.35, 1.0),
                     wavelength_range: Tuple[float, float] = (0.18, 0.90),
                     speed_range: Tuple[float, float] = (0.05, 0.25)) -> List[WaveComponent]:
    """
    Build a random pool of waves.
    The amplitude here is a multiplier. The global amplitude slider scales all waves.
    """
    waves = []
    for _ in range(num_waves):
        amp = float(rng.uniform(*amplitude_range))
        wavelength = float(rng.uniform(*wavelength_range))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        speed = float(rng.uniform(*speed_range))
        waves.append(WaveComponent(
            amplitude=amp,
            wavelength=wavelength,
            angle=angle,
            phase=phase,
            speed=speed
        ))
    return waves


def make_dynamic_height_field(H: int,
                              W: int,
                              xlim: Tuple[float, float],
                              ylim: Tuple[float, float],
                              waves: List[WaveComponent],
                              t: float,
                              z0: float = 0.0,
                              amplitude_scale: float = 1.0,
                              speed_scale: float = 1.0) -> HeightField:
    xmin, xmax = xlim
    ymin, ymax = ylim
    xs = np.linspace(xmin, xmax, W)
    ys = np.linspace(ymin, ymax, H)
    X, Y = np.meshgrid(xs, ys)

    h = np.zeros_like(X, dtype=np.float64)
    for w in waves:
        k = 2.0 * np.pi / w.wavelength
        kx = k * np.cos(w.angle)
        ky = k * np.sin(w.angle)
        omega = 2.0 * np.pi * (w.speed * speed_scale) / w.wavelength
        h += (w.amplitude * amplitude_scale) * np.sin(kx * X + ky * Y - omega * t + w.phase)

    return HeightField(h_map=h, xlim=xlim, ylim=ylim, z0=z0)


# ============================================================
# Camera and geometry
# ============================================================

def build_camera_rays(H: int,
                      W: int,
                      fx: float,
                      fy: float,
                      cx: float,
                      cy: float,
                      cam_pos=(0.0, 0.0, 1.0),
                      R: Optional[np.ndarray] = None):
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    x = (uu - cx) / fx
    y = (vv - cy) / fy

    rays_cam = np.stack([x, y, -np.ones_like(x)], axis=-1).astype(np.float64)
    rays_cam = normalize(rays_cam)

    if R is None:
        R = np.eye(3, dtype=np.float64)
    else:
        R = np.asarray(R, dtype=np.float64)

    rays_world = rays_cam @ R.T
    ray_o = np.zeros_like(rays_world) + np.array(cam_pos, dtype=np.float64)
    return ray_o, normalize(rays_world)


def intersect_flat_surface(ray_o: np.ndarray,
                           ray_d: np.ndarray,
                           z_water: float,
                           eps: float = 1e-10):
    dz = ray_d[..., 2]
    valid = np.abs(dz) > eps

    t = np.full(dz.shape, np.nan, dtype=np.float64)
    t[valid] = (z_water - ray_o[..., 2][valid]) / dz[valid]

    p = ray_o + np.nan_to_num(t, nan=0.0)[..., None] * ray_d
    return p, t


def intersect_height_field(ray_o: np.ndarray,
                           ray_d: np.ndarray,
                           surface: HeightField,
                           max_iter: int = 15,
                           tol: float = 1e-7,
                           eps: float = 1e-10):
    dz = ray_d[..., 2]
    valid0 = np.abs(dz) > eps

    t = np.full(dz.shape, np.nan, dtype=np.float64)
    t[valid0] = (surface.z0 - ray_o[..., 2][valid0]) / dz[valid0]

    finite_mask = np.isfinite(t)

    for _ in range(max_iter):
        if not np.any(finite_mask):
            break

        p = ray_o + np.nan_to_num(t, nan=0.0)[..., None] * ray_d
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        z_surf, _, hx, hy = surface.sample_height_and_normal(x, y)
        f = z - z_surf

        df = ray_d[..., 2] - hx * ray_d[..., 0] - hy * ray_d[..., 1]
        safe_df = np.abs(df) > eps
        update_mask = finite_mask & safe_df

        dt = np.zeros_like(t)
        dt[update_mask] = f[update_mask] / df[update_mask]

        t_new = t.copy()
        t_new[update_mask] = t[update_mask] - dt[update_mask]
        bad = ~np.isfinite(t_new)
        t_new[bad] = np.nan

        diff = np.abs(t_new - t)
        if np.nanmax(diff) < tol:
            t = t_new
            break

        t = t_new
        finite_mask = np.isfinite(t)

    p = ray_o + np.nan_to_num(t, nan=0.0)[..., None] * ray_d
    return p, t


def intersect_scene_plane(ray_o: np.ndarray,
                          ray_d: np.ndarray,
                          scene_z: float,
                          eps: float = 1e-10):
    dz = ray_d[..., 2]
    valid = np.abs(dz) > eps

    t = np.full(dz.shape, np.nan, dtype=np.float64)
    t[valid] = (scene_z - ray_o[..., 2][valid]) / dz[valid]
    p = ray_o + np.nan_to_num(t, nan=0.0)[..., None] * ray_d
    return p, t


# ============================================================
# Texture mapping
# ============================================================

def world_to_texture_coords(x: np.ndarray,
                            y: np.ndarray,
                            scene_bounds: Tuple[float, float, float, float],
                            tex_h: int,
                            tex_w: int):
    xmin, xmax, ymin, ymax = scene_bounds
    u = (x - xmin) / (xmax - xmin) * (tex_w - 1)
    v = (y - ymin) / (ymax - ymin) * (tex_h - 1)

    u = np.nan_to_num(u, nan=-1.0, posinf=-1.0, neginf=-1.0)
    v = np.nan_to_num(v, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return u.astype(np.float32), v.astype(np.float32)


# ============================================================
# Rendering
# ============================================================

def render_through_water(source_texture: np.ndarray,
                         image_size: Tuple[int, int],
                         fx: float,
                         fy: float,
                         cx: float,
                         cy: float,
                         cam_pos=(0.0, 0.0, 1.0),
                         scene_z: float = -1.0,
                         scene_bounds: Tuple[float, float, float, float] = (-1, 1, -1, 1),
                         water_surface: Optional[HeightField] = None,
                         water_level: float = 0.0,
                         n_air: float = 1.0,
                         n_water: float = 1.333,
                         return_debug: bool = False):
    H, W = image_size
    tex_h, tex_w = source_texture.shape[:2]

    ray_o, ray_d = build_camera_rays(H, W, fx, fy, cx, cy, cam_pos=cam_pos)

    p_ideal, t_ideal = intersect_scene_plane(ray_o, ray_d, scene_z)
    valid_ideal = np.isfinite(t_ideal) & (t_ideal > 0)

    if water_surface is None:
        p_int, t_int = intersect_flat_surface(ray_o, ray_d, water_level)
        normal = np.zeros_like(p_int)
        normal[..., 2] = 1.0
    else:
        p_int, t_int = intersect_height_field(ray_o, ray_d, water_surface)
        z_surf, normal, _, _ = water_surface.sample_height_and_normal(
            p_int[..., 0], p_int[..., 1]
        )
        p_int[..., 2] = z_surf

    valid = np.isfinite(t_int) & (t_int > 0)

    ray_w = refract(ray_d, normal, n_air, n_water)

    dz_scene = ray_w[..., 2]
    good_dz = np.abs(dz_scene) > 1e-10
    t_scene = np.full(dz_scene.shape, np.nan, dtype=np.float64)
    mask = valid & good_dz
    t_scene[mask] = (scene_z - p_int[..., 2][mask]) / dz_scene[mask]
    valid = valid & np.isfinite(t_scene) & (t_scene > 0)

    p_scene = p_int + np.nan_to_num(t_scene, nan=0.0)[..., None] * ray_w

    p_scene_x = np.where(valid, p_scene[..., 0], np.nan)
    p_scene_y = np.where(valid, p_scene[..., 1], np.nan)

    map_x, map_y = world_to_texture_coords(
        p_scene_x, p_scene_y, scene_bounds, tex_h, tex_w
    )

    if source_texture.ndim == 2:
        rendered = cv2.remap(
            source_texture, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    else:
        channels = []
        for c in range(source_texture.shape[2]):
            ch = cv2.remap(
                source_texture[..., c], map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            channels.append(ch)
        rendered = np.stack(channels, axis=-1)

    if not return_debug:
        return rendered

    return {
        "frame": rendered,
        "valid": valid,
        "map_x": map_x,
        "map_y": map_y,
        "p_scene": p_scene,
        "p_ideal": p_ideal,
        "valid_ideal": valid_ideal,
    }


# ============================================================
# Diagnostics
# ============================================================

def radial_distortion_test_from_world(p_ideal: np.ndarray,
                                      p_distorted: np.ndarray,
                                      valid: np.ndarray,
                                      eps: float = 1e-12):
    xi = p_ideal[..., 0]
    yi = p_ideal[..., 1]
    xd = p_distorted[..., 0]
    yd = p_distorted[..., 1]

    dx = xd - xi
    dy = yd - yi

    r = np.sqrt(xi**2 + yi**2)
    disp = np.sqrt(dx**2 + dy**2)

    cross = xi * dy - yi * dx
    radial_error = np.abs(cross) / np.maximum(r * np.maximum(disp, eps), eps)

    tiny_disp = disp < 1e-10
    radial_error[tiny_disp] = 0.0

    radial_error[~valid] = np.nan

    mean_error = np.nanmean(radial_error)
    max_error = np.nanmax(radial_error)
    return radial_error, mean_error, max_error


def make_error_heatmap(error_map: np.ndarray, title: str = "") -> np.ndarray:
    err = np.nan_to_num(error_map, nan=0.0)
    m = float(np.max(err)) if np.size(err) else 0.0
    if m > 0:
        vis = (255.0 * err / m).astype(np.uint8)
    else:
        vis = np.zeros(err.shape, dtype=np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    if title:
        vis = overlay_text(vis, [title], x=10, y=20)
    return vis


def make_vector_field_vis(p_ideal: np.ndarray,
                          p_distorted: np.ndarray,
                          valid: np.ndarray,
                          step: int = 25,
                          canvas_size: Tuple[int, int] = (600, 800),
                          scale: float = 120.0) -> np.ndarray:
    H, W = canvas_size
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    for y in range(0, H, step):
        for x in range(0, W, step):
            if not valid[y, x]:
                continue
            xi, yi = p_ideal[y, x, 0], p_ideal[y, x, 1]
            xd, yd = p_distorted[y, x, 0], p_distorted[y, x, 1]
            dx = xd - xi
            dy = yd - yi

            p0 = (int(x), int(y))
            p1 = (int(round(x + scale * dx)), int(round(y + scale * dy)))
            cv2.arrowedLine(canvas, p0, p1, (0, 60, 220), 1, tipLength=0.3)

    cv2.putText(canvas, "Distortion vector field", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def fit_brown_conrady_world(p_ideal: np.ndarray,
                            p_distorted: np.ndarray,
                            valid: np.ndarray,
                            max_radius_quantile: float = 0.98):
    xi = p_ideal[..., 0][valid]
    yi = p_ideal[..., 1][valid]
    xd = p_distorted[..., 0][valid]
    yd = p_distorted[..., 1][valid]

    ri = np.sqrt(xi**2 + yi**2)
    rd = np.sqrt(xd**2 + yd**2)

    good = (ri > 1e-8) & np.isfinite(ri) & np.isfinite(rd)
    ri = ri[good]
    rd = rd[good]

    if len(ri) < 20:
        return None

    rmax = np.quantile(ri, max_radius_quantile)
    use = ri <= rmax
    ri = ri[use]
    rd = rd[use]

    s = rd / np.maximum(ri, 1e-12)
    A = np.stack([ri**2, ri**4, ri**6], axis=1)
    b = s - 1.0

    k, *_ = np.linalg.lstsq(A, b, rcond=None)
    k1, k2, k3 = k

    s_fit = 1.0 + k1 * ri**2 + k2 * ri**4 + k3 * ri**6
    rd_fit = ri * s_fit

    mae = np.mean(np.abs(rd_fit - rd))
    rmse = math.sqrt(np.mean((rd_fit - rd)**2))

    return {
        "k1": float(k1),
        "k2": float(k2),
        "k3": float(k3),
        "mae": float(mae),
        "rmse": float(rmse),
        "ri": ri,
        "rd": rd,
        "rd_fit": rd_fit,
    }


def make_radial_fit_plot(fit_result,
                         size: Tuple[int, int] = (420, 700)) -> np.ndarray:
    H, W = size
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    if fit_result is None:
        return overlay_text(canvas, ["Not enough valid points for fit"], x=20, y=40)

    ri = fit_result["ri"]
    rd = fit_result["rd"]
    rd_fit = fit_result["rd_fit"]

    margin = 50
    plot_w = W - 2 * margin
    plot_h = H - 2 * margin

    x_max = max(float(np.max(ri)), 1e-6)
    y_max = max(float(np.max(np.concatenate([rd, rd_fit]))), 1e-6)

    def to_pt(x, y):
        px = int(margin + plot_w * (x / x_max))
        py = int(H - margin - plot_h * (y / y_max))
        return px, py

    cv2.line(canvas, (margin, H - margin), (W - margin, H - margin), (0, 0, 0), 1)
    cv2.line(canvas, (margin, H - margin), (margin, margin), (0, 0, 0), 1)

    idx = np.linspace(0, len(ri) - 1, min(1200, len(ri))).astype(int)
    for i in idx:
        cv2.circle(canvas, to_pt(ri[i], rd[i]), 1, (180, 180, 180), -1)

    order = np.argsort(ri)
    pts = [to_pt(float(ri[i]), float(rd_fit[i])) for i in order]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(canvas, a, b, (0, 0, 255), 2)

    lines = [
        "Brown-Conrady radial fit on world-plane radius",
        f"k1={fit_result['k1']:.6e}",
        f"k2={fit_result['k2']:.6e}",
        f"k3={fit_result['k3']:.6e}",
        f"MAE={fit_result['mae']:.6e}",
        f"RMSE={fit_result['rmse']:.6e}",
        "gray: measured  red: fitted"
    ]
    return overlay_text(canvas, lines, x=15, y=22)


def amplitude_sweep_plot(source_texture: np.ndarray,
                         image_size: Tuple[int, int],
                         fx: float,
                         fy: float,
                         cx: float,
                         cy: float,
                         cam_pos,
                         scene_z: float,
                         scene_bounds,
                         water_level: float,
                         n_air: float,
                         n_water: float,
                         waves: List[WaveComponent],
                         water_grid_H: int,
                         water_grid_W: int,
                         water_xylim,
                         t: float,
                         speed_scale: float,
                         max_amp_scale: float,
                         n_samples: int = 12,
                         size: Tuple[int, int] = (420, 700)) -> np.ndarray:
    amps = np.linspace(0.0, max_amp_scale, n_samples)
    errs = []

    diag_H, diag_W = 220, 300
    diag_fx = fx * (diag_W / image_size[1])
    diag_fy = fy * (diag_H / image_size[0])
    diag_cx = diag_W / 2.0
    diag_cy = diag_H / 2.0

    small_tex = cv2.resize(source_texture, (600, 450), interpolation=cv2.INTER_AREA)

    for a in amps:
        surface = make_dynamic_height_field(
            H=water_grid_H,
            W=water_grid_W,
            xlim=water_xylim[0],
            ylim=water_xylim[1],
            waves=waves,
            t=t,
            z0=water_level,
            amplitude_scale=float(a),
            speed_scale=float(speed_scale),
        )
        dbg = render_through_water(
            source_texture=small_tex,
            image_size=(diag_H, diag_W),
            fx=diag_fx, fy=diag_fy, cx=diag_cx, cy=diag_cy,
            cam_pos=cam_pos,
            scene_z=scene_z,
            scene_bounds=scene_bounds,
            water_surface=surface,
            water_level=water_level,
            n_air=n_air,
            n_water=n_water,
            return_debug=True
        )
        valid = dbg["valid"] & dbg["valid_ideal"]
        _, mean_err, _ = radial_distortion_test_from_world(
            dbg["p_ideal"], dbg["p_scene"], valid
        )
        errs.append(float(mean_err))

    amps = np.array(amps)
    errs = np.array(errs)

    H, W = size
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    margin = 50
    plot_w = W - 2 * margin
    plot_h = H - 2 * margin

    x_max = max(float(np.max(amps)), 1e-6)
    y_max = max(float(np.max(errs)), 1e-9)

    def to_pt(x, y):
        px = int(margin + plot_w * (x / x_max))
        py = int(H - margin - plot_h * (y / y_max))
        return px, py

    cv2.line(canvas, (margin, H - margin), (W - margin, H - margin), (0, 0, 0), 1)
    cv2.line(canvas, (margin, H - margin), (margin, margin), (0, 0, 0), 1)

    pts = [to_pt(float(a), float(e)) for a, e in zip(amps, errs)]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(canvas, a, b, (255, 0, 0), 2)
    for p in pts:
        cv2.circle(canvas, p, 3, (0, 0, 255), -1)

    lines = [
        "Mean non-radial error vs wave amplitude scale",
        f"current n_water={n_water:.3f}, focal={fx:.1f}, depth={abs(scene_z - water_level):.2f}",
        f"max amplitude scale={max_amp_scale:.3f}",
    ]
    return overlay_text(canvas, lines, x=15, y=22)


# ============================================================
# User image helpers
# ============================================================

def make_demo_texture(tex_h: int, tex_w: int) -> np.ndarray:
    img = np.zeros((tex_h, tex_w, 3), dtype=np.uint8)

    for y in range(tex_h):
        img[y, :, 0] = np.clip(50 + 150 * y / tex_h, 0, 255)
        img[y, :, 1] = np.clip(80 + 100 * (1 - y / tex_h), 0, 255)
        img[y, :, 2] = 140

    for x in range(0, tex_w, 50):
        cv2.line(img, (x, 0), (x, tex_h - 1), (255, 255, 255), 1)
    for y in range(0, tex_h, 50):
        cv2.line(img, (0, y), (tex_w - 1, y), (255, 255, 255), 1)

    for r in [40, 80, 120, 160]:
        cv2.circle(img, (tex_w // 2, tex_h // 2), r, (0, 0, 255), 2)

    cv2.putText(img, "UNDERWATER PLANE", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, "UNDERWATER PLANE", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def center_crop_to_aspect(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    aspect = w / h

    if aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        cropped = img[:, x0:x0 + new_w]
    else:
        new_h = int(round(w / target_aspect))
        y0 = (h - new_h) // 2
        cropped = img[y0:y0 + new_h, :]

    return cropped


def load_user_image(path: Optional[str], target_size: Tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size

    if path is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Choose an image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                    ("Image files", "*.PNG *.JPG *.JPEG *.BMP *.TIF *.TIFF *.WEBP"),
                    ("All files", "*.*"),
                ],
            )
            root.destroy()
        except Exception:
            path = None

    if path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not load image: {path}")
        else:
            print(f"Loaded image: {path}")
            img = center_crop_to_aspect(img, target_w, target_h)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return img

    print("Using fallback demo texture.")
    return make_demo_texture(target_h, target_w)


# ============================================================
# Main interactive demo
# ============================================================

def main():
    H, W = 600, 800
    cx, cy = W / 2.0, H / 2.0

    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    source_texture = load_user_image(image_path, target_size=(1200, 900))

    water_level = 0.0
    scene_bounds = (-1.2, 1.2, -0.9, 0.9)

    water_grid_H, water_grid_W = 220, 260
    water_xylim = ((-1.4, 1.4), (-1.1, 1.1))

    max_wave_pool = 40
    rng = np.random.default_rng()
    wave_pool = random_wave_pool(max_wave_pool, rng)

    window = "Dynamic Water Refraction"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1200, 850)

    # Core sliders
    cv2.createTrackbar("camera_height_x100", window, 80, 300, nothing)
    cv2.createTrackbar("water_depth_x100",  window, 80, 300, nothing)
    cv2.createTrackbar("focal_x10",         window, 7000, 20000, nothing)
    cv2.createTrackbar("wave_amp_x1000",    window, 100, 400, nothing)
    cv2.createTrackbar("wave_speed_x100",   window, 100, 400, nothing)
    cv2.createTrackbar("n_water_x100",      window, 133, 1000, nothing)
    cv2.createTrackbar("num_waves",         window, 3, max_wave_pool, nothing)
    cv2.createTrackbar("show_mask",         window, 0, 1, nothing)
    cv2.createTrackbar("pause",             window, 0, 1, nothing)

    # Diagnostics
    cv2.createTrackbar("diag_radial_test",  window, 0, 1, nothing)
    cv2.createTrackbar("diag_vectors",      window, 0, 1, nothing)
    cv2.createTrackbar("diag_fit",          window, 0, 1, nothing)
    cv2.createTrackbar("diag_amp_plot",     window, 0, 1, nothing)

    sim_time = 0.0
    last_tick = cv2.getTickCount()
    tick_freq = cv2.getTickFrequency()

    last_amp_plot_tick = 0
    cached_amp_plot = None
    cached_amp_params = None

    print("Press ESC to quit.")
    print("Press R to randomize the wave pool.")

    while True:
        current_tick = cv2.getTickCount()
        dt = (current_tick - last_tick) / tick_freq
        last_tick = current_tick

        paused = cv2.getTrackbarPos("pause", window) == 1
        if not paused:
            sim_time += dt

        camera_height = get_slider_float(window, "camera_height_x100", 100.0)
        water_depth   = get_slider_float(window, "water_depth_x100", 100.0)
        focal         = get_slider_float(window, "focal_x10", 10.0)
        amp_scale     = get_slider_float(window, "wave_amp_x1000", 1000.0)
        speed_scale   = get_slider_float(window, "wave_speed_x100", 100.0)
        n_water       = get_slider_float(window, "n_water_x100", 100.0)
        num_waves     = cv2.getTrackbarPos("num_waves", window)

        show_mask   = cv2.getTrackbarPos("show_mask", window) == 1
        do_radial   = cv2.getTrackbarPos("diag_radial_test", window) == 1
        do_vectors  = cv2.getTrackbarPos("diag_vectors", window) == 1
        do_fit      = cv2.getTrackbarPos("diag_fit", window) == 1
        do_amp_plot = cv2.getTrackbarPos("diag_amp_plot", window) == 1

        camera_height = max(camera_height, 0.05)
        water_depth = max(water_depth, 0.05)
        focal = max(focal, 50.0)
        n_water = max(n_water, 1.0)
        num_waves = max(num_waves, 0)

        cam_pos = (0.0, 0.0, camera_height)
        scene_z = water_level - water_depth

        active_waves = wave_pool[:num_waves]

        surface = make_dynamic_height_field(
            H=water_grid_H,
            W=water_grid_W,
            xlim=water_xylim[0],
            ylim=water_xylim[1],
            waves=active_waves,
            t=sim_time,
            z0=water_level,
            amplitude_scale=amp_scale,
            speed_scale=speed_scale
        )

        dbg = render_through_water(
            source_texture=source_texture,
            image_size=(H, W),
            fx=focal, fy=focal,
            cx=cx, cy=cy,
            cam_pos=cam_pos,
            scene_z=scene_z,
            scene_bounds=scene_bounds,
            water_surface=surface,
            water_level=water_level,
            n_air=1.0,
            n_water=n_water,
            return_debug=True
        )

        frame = dbg["frame"]
        valid = dbg["valid"]
        valid_joint = dbg["valid"] & dbg["valid_ideal"]

        if show_mask:
            mask_vis = (valid.astype(np.uint8) * 255)
            mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 0.8, mask_vis, 0.2, 0.0)

        mean_err = float("nan")
        max_err = float("nan")
        fit_result = None

        if do_radial or do_fit or do_vectors:
            error_map, mean_err, max_err = radial_distortion_test_from_world(
                dbg["p_ideal"], dbg["p_scene"], valid_joint
            )

        if do_radial:
            heat = make_error_heatmap(
                error_map,
                title="Non-radial error heatmap"
            )
            cv2.imshow("radial_error_heatmap", heat)
        else:
            try:
                cv2.destroyWindow("radial_error_heatmap")
            except cv2.error:
                pass

        if do_vectors:
            vec = make_vector_field_vis(
                dbg["p_ideal"], dbg["p_scene"], valid_joint,
                step=28, canvas_size=(H, W), scale=120.0
            )
            cv2.imshow("distortion_vectors", vec)
        else:
            try:
                cv2.destroyWindow("distortion_vectors")
            except cv2.error:
                pass

        if do_fit:
            fit_result = fit_brown_conrady_world(
                dbg["p_ideal"], dbg["p_scene"], valid_joint
            )
            fit_plot = make_radial_fit_plot(fit_result)
            cv2.imshow("brown_conrady_fit", fit_plot)
        else:
            try:
                cv2.destroyWindow("brown_conrady_fit")
            except cv2.error:
                pass

        if do_amp_plot:
            param_key = (
                round(camera_height, 3),
                round(water_depth, 3),
                round(focal, 1),
                round(speed_scale, 3),
                round(n_water, 3),
                round(amp_scale, 3),
                round(sim_time, 1),
                int(num_waves),
                tuple((round(w.amplitude, 4),
                       round(w.wavelength, 4),
                       round(w.angle, 4),
                       round(w.phase, 4),
                       round(w.speed, 4)) for w in active_waves),
            )
            now = cv2.getTickCount()
            elapsed = (now - last_amp_plot_tick) / tick_freq

            if cached_amp_plot is None or cached_amp_params != param_key or elapsed > 0.7:
                cached_amp_plot = amplitude_sweep_plot(
                    source_texture=source_texture,
                    image_size=(H, W),
                    fx=focal, fy=focal, cx=cx, cy=cy,
                    cam_pos=cam_pos,
                    scene_z=scene_z,
                    scene_bounds=scene_bounds,
                    water_level=water_level,
                    n_air=1.0,
                    n_water=n_water,
                    waves=active_waves,
                    water_grid_H=120,
                    water_grid_W=140,
                    water_xylim=water_xylim,
                    t=sim_time,
                    speed_scale=speed_scale,
                    max_amp_scale=max(amp_scale, 0.001),
                    n_samples=12,
                )
                cached_amp_params = param_key
                last_amp_plot_tick = now

            cv2.imshow("amplitude_sweep", cached_amp_plot)
        else:
            try:
                cv2.destroyWindow("amplitude_sweep")
            except cv2.error:
                pass

        fit_line = "fit: off"
        if fit_result is not None:
            fit_line = f"k1={fit_result['k1']:.2e} k2={fit_result['k2']:.2e} k3={fit_result['k3']:.2e}"

        info_lines = [
            f"camera height: {camera_height:.2f}",
            f"water depth:   {water_depth:.2f}",
            f"focal length:  {focal:.1f}",
            f"n_water:       {n_water:.3f}",
            f"num waves:     {num_waves}",
            f"wave amp scl:  {amp_scale:.3f}",
            f"wave spd scl:  {speed_scale:.2f}",
            f"mean non-radial err: {mean_err:.3e}" if not np.isnan(mean_err) else "mean non-radial err: off",
            f"max non-radial err:  {max_err:.3e}" if not np.isnan(max_err) else "max non-radial err: off",
            fit_line,
            "Tip: set num_waves=0 or wave_amp_scl=0 for flat-water test",
            "ESC = quit   |   R = randomize wave pool"
        ]
        frame = overlay_text(frame, info_lines)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key in (ord('r'), ord('R')):
            wave_pool = random_wave_pool(max_wave_pool, rng)
            cached_amp_plot = None
            cached_amp_params = None
            print("Randomized wave pool.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()