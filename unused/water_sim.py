import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# --- Water & refraction parameters ---
n_air, n_water = 1.0, 1.333
depth = 10
enable_waves = False  # Set False for flat water

# --- SIFT detector ---
sift = cv2.SIFT_create(nfeatures=300)

# --- Wave simulation ---
class WaveField:
    def __init__(self, w, h, num_waves=20):
        self.w, self.h = w, h
        rng = np.random.default_rng()
        self.waves = []
        for _ in range(num_waves):
            wavelength = rng.uniform(40,150)
            amplitude = rng.uniform(1,6)
            speed = rng.uniform(0.2,1.0)
            theta = rng.uniform(0,2*np.pi)
            phase = rng.uniform(0,2*np.pi)
            self.waves.append((amplitude, wavelength, speed, theta, phase))
        y, x = np.mgrid[0:h,0:w]
        self.x, self.y = x, y

    def height(self, t):
        if not enable_waves:
            return np.zeros((self.h, self.w), np.float32)
        H = np.zeros((self.h, self.w), np.float32)
        for A,L,s,theta,phi in self.waves:
            k = 2*np.pi/L
            dx, dy = np.cos(theta), np.sin(theta)
            phase = k*(dx*self.x + dy*self.y) - s*t + phi
            H += A*np.sin(phase)
        return H

# --- Displacement from waves ---
def compute_wave_displacement(H):
    dHx = np.gradient(H, axis=1)
    dHy = np.gradient(H, axis=0)
    factor = depth*(1-n_air/n_water)
    return dHx*factor, dHy*factor

# --- Flat water radial refraction ---
def flat_water_displacement(h, w, depth, n_air=1.0, n_water=1.333):
    y, x = np.mgrid[0:h,0:w].astype(np.float32)
    cx, cy = w/2, h/2
    rx = x - cx
    ry = y - cy
    r = np.sqrt(rx**2 + ry**2) + 1e-7
    scale = (1 - n_air/n_water) * depth / r
    dx = rx * scale
    dy = ry * scale
    return dx, dy

# --- Combine wave + flat displacement ---
def compute_total_displacement(H, h, w):
    dx_flat, dy_flat = flat_water_displacement(h, w, depth)
    dx_wave, dy_wave = compute_wave_displacement(H)
    dx_total = dx_flat + dx_wave
    dy_total = dy_flat + dy_wave
    return dx_total, dy_total

# --- Distort image ---
def distort(img, dx, dy):
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h,0:w].astype(np.float32)
    return cv2.remap(img, x+dx, y+dy, cv2.INTER_LINEAR)

# --- SRD-SIFT descriptors ---
def compute_srd_sift_descriptors(orig_gray, keypoints, dx, dy):
    descriptors = []
    h, w = orig_gray.shape
    for kp in keypoints:
        x, y = kp.pt
        x_u = np.clip(x - dx[int(round(y)), int(round(x))] + np.random.uniform(-1,1), 0, w-1)
        y_u = np.clip(y - dy[int(round(y)), int(round(x))] + np.random.uniform(-1,1), 0, h-1)
        patch_size = 16
        r = patch_size//2
        x0, y0 = int(max(x_u-r,0)), int(max(y_u-r,0))
        x1, y1 = int(min(x_u+r,w-1)), int(min(y_u+r,h-1))
        patch = orig_gray[y0:y1, x0:x1]
        if patch.size < 16*16:
            descriptors.append(np.zeros(128,np.float32))
            continue
        temp_kp = cv2.KeyPoint((x1-x0)/2.0,(y1-y0)/2.0,float(patch_size))
        _, desc = sift.compute(patch,[temp_kp])
        if desc is not None:
            desc = desc[0].astype(np.float32)
            desc /= np.linalg.norm(desc)+1e-7
            descriptors.append(desc)
        else:
            descriptors.append(np.zeros(128,np.float32))
    return np.array(descriptors)

# --- Load image ---
img = cv2.imread("input.jpg")
if img is None:
    raise FileNotFoundError("Place 'input.jpg' in the folder")
h, w = img.shape[:2]
waves = WaveField(w,h)

# --- Reference keypoints and descriptors ---
gray_ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kp_ref = sift.detect(gray_ref,None)
_, desc_ref = sift.compute(gray_ref,kp_ref)
ref_pts = np.array([kp.pt for kp in kp_ref])
tree = cKDTree(ref_pts)

# --- Live plot setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(6,4))
bars = ax.bar(['Standard SIFT','SRD-SIFT'], [0,0], color=['blue','orange'])
ax.set_ylim(0,1)
ax.set_ylabel("Matching Accuracy")
ax.set_title("Live SIFT vs SRD-SIFT Accuracy")
fig.canvas.draw()
fig.canvas.flush_events()

# --- Prepare undistortion accumulators ---
undistorted_accum = np.zeros_like(img, dtype=np.float32)
weight_accum = np.zeros((h, w), dtype=np.float32)

start = time.time()

while True:
    t = time.time()-start
    H = waves.height(t)
    dx, dy = compute_total_displacement(H, h, w)
    distorted = distort(img, dx, dy)
    gray = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)

    # --- Flatten for accumulation ---
    y_coords, x_coords = np.mgrid[0:h,0:w].astype(np.float32)
    x_u = x_coords - dx
    y_u = y_coords - dy

    x_flat_1d = np.clip(np.floor(x_u).astype(int).ravel(),0,w-1)
    y_flat_1d = np.clip(np.floor(y_u).astype(int).ravel(),0,h-1)
    img_flat = distorted.reshape(-1,3)

    flat_idx = np.ravel_multi_index((y_flat_1d, x_flat_1d), (h, w))

    # --- Accumulate per channel ---
    undistorted_flat_r = undistorted_accum[:,:,0].ravel()
    undistorted_flat_g = undistorted_accum[:,:,1].ravel()
    undistorted_flat_b = undistorted_accum[:,:,2].ravel()
    weight_flat = weight_accum.ravel()

    np.add.at(undistorted_flat_r, flat_idx, img_flat[:,0])
    np.add.at(undistorted_flat_g, flat_idx, img_flat[:,1])
    np.add.at(undistorted_flat_b, flat_idx, img_flat[:,2])
    np.add.at(weight_flat, flat_idx, 1)

    undistorted_accum[:,:,0] = undistorted_flat_r.reshape(h,w)
    undistorted_accum[:,:,1] = undistorted_flat_g.reshape(h,w)
    undistorted_accum[:,:,2] = undistorted_flat_b.reshape(h,w)
    weight_accum = weight_flat.reshape(h,w)

    # --- Detect keypoints ---
    kp_std = sift.detect(gray,None)
    _, desc_std = sift.compute(gray,kp_std) if kp_std else ([], None)

    # --- Standard SIFT accuracy ---
    acc_std = 0
    if desc_std is not None and len(desc_std)>0:
        correct = 0
        for i,kp in enumerate(kp_std):
            x,y = kp.pt
            x_u = np.clip(x - dx[int(round(y)), int(round(x))],0,w-1)
            y_u = np.clip(y - dy[int(round(y)), int(round(x))],0,h-1)
            dist, idx = tree.query([x_u, y_u])
            d_ref = desc_ref[idx]
            d_curr = desc_std[i]
            if np.linalg.norm(d_curr - d_ref) < 0.7:
                correct += 1
        acc_std = correct / len(desc_std)

    # --- SRD-SIFT accuracy ---
    acc_srd = 0
    if desc_std is not None and len(desc_std)>0:
        desc_srd = compute_srd_sift_descriptors(gray_ref, kp_std, dx, dy)
        correct = 0
        for i,kp in enumerate(kp_std):
            x,y = kp.pt
            x_u = np.clip(x - dx[int(round(y)), int(round(x))],0,w-1)
            y_u = np.clip(y - dy[int(round(y)), int(round(x))],0,h-1)
            dist, idx = tree.query([x_u, y_u])
            d_ref = desc_ref[idx]
            d_curr = desc_srd[i]
            if np.linalg.norm(d_curr - d_ref) < 0.7:
                correct += 1
        acc_srd = correct / len(desc_std)

    # --- Update live bar chart ---
    bars[0].set_height(acc_std)
    bars[1].set_height(acc_srd)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # --- Show frame ---
    frame_vis = cv2.drawKeypoints(distorted,kp_std,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("SIFT vs SRD-SIFT",frame_vis)
    if cv2.waitKey(16)==27:
        break

cv2.destroyAllWindows()
plt.ioff()
plt.show()

# --- Compute final undistorted image ---
undistorted_final = undistorted_accum / np.maximum(weight_accum[:,:,None],1)
undistorted_final = np.clip(undistorted_final,0,255).astype(np.uint8)

cv2.imshow("Undistorted Average Image", undistorted_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
