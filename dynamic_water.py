import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Waves():

    def __init__(self):
        self.amplitudes = []
        self.directions = []
        self.ang_frequencies = []
        self.phases = []

    def amps(self):
        return np.array(self.amplitudes)[:, np.newaxis, np.newaxis]
    
    def dir(self):
        return np.array(self.directions)
    
    def ang_freq(self):
        return np.array(self.ang_frequencies)[:, np.newaxis, np.newaxis]

    def phs(self):
        return np.array(self.phases)[:, np.newaxis, np.newaxis]
    
    def indices(self, field_shape, r):
        yy, xx = np.indices(field_shape, dtype=np.float64)
        size = max(np.max(xx), np.max(yy))
        xx = r * xx / size
        yy = r * yy / size
        return np.stack((yy,xx))

    def add_wave(self, amp, dir, ang, phs):
        self.amplitudes.append(amp)
        self.directions.append(dir)
        self.ang_frequencies.append(ang)
        self.phases.append(phs)

    def generate_wave(self, a_amp, s_amp, h, f, r):

        pixel_to_world = (h / f)

        lam_px = np.random.uniform(0.1 * r, r)
        lam_world = lam_px * pixel_to_world
        

        amp = np.random.normal(a_amp, s_amp, (1))[0]
        k_mag = 2 * np.pi / lam_world

        # --- direction ---
        theta = np.random.uniform(0, 2*np.pi)
        kx = k_mag * np.cos(theta)
        ky = k_mag * np.sin(theta)
        dir = np.array([kx, ky])

        # --- frequency (deep water dispersion) ---
        speed = np.random.uniform(0.2, 10)
        omega = np.sqrt(speed * k_mag)

        # --- phase ---
        phi = np.random.uniform(0, 2*np.pi)

        self.add_wave(amp, dir, omega, phi)

    def generate_n_waves(self, n, a_amp, s_amp, h ,f, r):
        for i in range(n):
            self.generate_wave(a_amp, s_amp, h, f, r)

    def height_field(self, field_shape, t, r):
        directions = np.einsum('k i, i n m -> k n m', self.dir(), self.indices(field_shape, r))
        heights = self.amps() * np.sin(directions - self.ang_freq() * t + self.phs())
        hf = np.sum(heights, axis = 0)
        return hf
    
    def grad_field(self, field_shape, t, r):
        directions = np.einsum('k i, i n m -> k n m', self.dir(), self.indices(field_shape, r))
        grad = self.amps() * np.sin(directions - self.ang_freq() * t + self.phs())

        coeff_x = self.dir()[:, 1][:, np.newaxis]
        coeff_y = self.dir()[:, 0][:, np.newaxis]
        grad_x = np.sum(coeff_x * grad, axis = 0)
        grad_y = np.sum(coeff_y * grad, axis = 0)
        up = np.ones_like(grad_x)

        grads =  np.stack((grad_x, grad_y, up))
        norms = np.linalg.norm(grads, axis = 0)
        grads = grads / norms

        return grads

class Surface():
    
    def __init__(self, image):
        self.image = image
        self.h_w = 1
        self.h_p = 1
        self.f = 1
        self.waves = Waves()
        self.height_field = [np.zeros(image.shape)]
        self.gradient_field = [np.zeros(((3, ) + image.shape))]
        self.dt = 0.1
        self.t = 0
    
    def compute_height_field(self):
        if len(self.waves == 0):
            hf = np.zeros_like(self.height_field[0])
            self.height_field.append(hf)
            return hf
        

waves = Waves()
waves.generate_n_waves(3, 1, 0.3, 20, 20, 1)
data = ((waves.height_field((1000, 1000), 0.0, 5)))


fig, ax = plt.subplots()
im = ax.matshow(data, cmap='viridis', vmin=0, vmax=1)
fig.colorbar(im)

T = 10
dt = 0.1
times = np.arange(0, T, dt)

def update(time):
    im.set_array(((waves.height_field((1000, 1000), time, 5)))) # Update data
    return [im]

ani = animation.FuncAnimation(fig, update, frames=times, interval=100, blit=True)
plt.show()