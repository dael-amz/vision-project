import cv2
from water_surface_simulator import WaterSurfaceSimulator

from water_surface_simulator import WaterSurfaceSimulator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


img = cv2.imread("input.jpg")

sim = WaterSurfaceSimulator(
    image=img,
    h=5.0,
    d=1.0,
    f=20.0,
    dt=0.04,
    duration=3.0,
    n_waves=8,
    wavelength_px_range=(25, 140),
    amplitude_range=(0.002, 0.015),
    seed=0,
)

frames = sim.generate_frames()
print(len(frames))
sim.save_video("distorted.mp4")


#img = cv2.imread("input.jpg")

#sim = WaterSurfaceSimulator(image=img, h=1.0, d=1.0, f=500.0, seed=0)

dist_func = sim.make_distortion_func(t=0.5)

origin_loc = np.array([[120.0, 80.0], [250.0, 200.0], [140.0, 200.0], [10, 2]])
origin_scales = np.array([3.0, 5.0, 1, 10])

distorted_loc, distorted_scales = dist_func(origin_loc, origin_scales)

fig, ax = plt.subplots()
scat1 = ax.scatter(origin_loc[:,0], origin_loc[:,1], origin_scales ,color='blue', label='Set 1')
scat2 = ax.scatter(distorted_loc[:,0], distorted_loc[:,1], distorted_scales, color='red', label='Set 2')


T = 10
dt = 0.1
times = np.arange(0, T, dt)

def update(time):
    dist_func = sim.make_distortion_func(t=time)
    distorted_loc, distorted_scales = dist_func(origin_loc, origin_scales)
    #scat1.set_offsets(points1[frame])
    scat2.set_offsets(distorted_loc)
    return scat1, scat2

ani = animation.FuncAnimation(fig, update, frames=times, interval=100, blit=True)
plt.show()
