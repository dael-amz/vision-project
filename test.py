import math
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.ndimage as ndi
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian
from scipy.stats import multivariate_normal
import sift
from numba import jit
from skimage.transform import rescale
from tqdm import tqdm
from skimage._shared.filters import gaussian


image = cv2.imread("input.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

upsampling = 1
n_scales = 3
n_octaves = 8
sigma_min = 1.6
sigma_in = 0.5
deltas = (1 / upsampling) * np.power(
            2, np.arange(n_octaves), dtype=None
        )


def _create_1d_gaussians(image_shape, xi, center = None, truncate = 4.0):

        gaussians = {}

        h, w = image_shape

        # match the image size used by _create_scalespace
        # if upsampling > 1:
        #     h = int(round(h * upsampling))
        #     w = int(round(w * upsampling))

        if center is None:
            cx0 = (w - 1) / 2.0
            cy0 = (h - 1) / 2.0
        else:
            cx0 = center[0] * upsampling
            cy0 = center[1] * upsampling

        # same sigma bookkeeping as skimage SIFT
        tmp = np.power(2, np.arange(n_scales + 3) / n_scales) * sigma_min
        sigmas = deltas[:, np.newaxis] / deltas[0] * tmp[np.newaxis, :]
        scalespace_sigmas = (
                deltas[:, np.newaxis] / deltas[0]
            ) * tmp[np.newaxis, :]

        # for exact-paper construction from octave base:
        # use blur-from-base sigmas, not incremental sigmas
        sigma_from_base = np.zeros_like(sigmas)
        sigma_from_base[:, 0] = upsampling * math.sqrt(
            sigma_min**2 - sigma_in**2
        )

        var_diff = np.diff(sigmas * sigmas, axis=1)
        gaussian_sigmas = np.sqrt(var_diff) / deltas[:, np.newaxis]

        for o in range(n_octaves):
            for s in range(1, n_scales + 3):
                sigma_from_base[o, s] = math.sqrt(
                    max(sigmas[o, s] ** 2 - sigmas[o, 0] ** 2, 0.0)
                )

        def kernel1d(s):
            #print("S", float(s))
            s = max(float(s), 1e-12)
            rad = max(1, int(truncate * s + 0.5))
            x = np.arange(-rad, rad + 1, dtype=None)
            #print(s)
            k = np.exp(-(x * x) / (2.0 * s * s))
            return k / k.sum()

        kernels = {}
        flat_kernels = {}
        r2_maps = []

        for o in range(n_octaves):

            ho = math.ceil(h / (2**o))
            wo = math.ceil(w / (2**o))
            cxo = cx0 / (2**o)
            cyo = cy0 / (2**o)

            yy, xx = np.indices((ho, wo), dtype=None)
            r2_map = np.round((xx - cxo) ** 2 + (yy - cyo) ** 2).astype(np.int64)
            r2_maps.append(r2_map)

            unique_r2 = np.unique(r2_map)
            rad = max(unique_r2)
            kernels[o] = {}

            for s in gaussian_sigmas[o]:
                #print("WTH", s)

                s = float(s)
                kernels[o][s] = {}
                flat_kernels.setdefault(s, {})

                for r2 in unique_r2:
                    r2 = (r2 / rad)
                    if r2 not in flat_kernels[s]:
                        sigma_r = s * abs(1.0 + xi * r2)
                        #print("Sigma_r", sigma_r)
                        flat_kernels[s][r2] = kernel1d(sigma_r)
                    kernels[o][s][r2] = flat_kernels[s][r2]


        # key = list(kernels[0].keys())[3]
        # key2 = list(kernels[0][key].keys())[-1]
        # #print(len(kernels[0][key]))
        # plt.plot(kernels[0][key][key2])
        # plt.show()
        #return 1
        return flat_kernels, kernels, r2_maps, sigma_from_base


def _apply_kernels(kernels, image, r2_maps, idx, granularity):

    octave = np.empty(
            (n_scales + 3,) + image.shape, dtype=None, order='C'
        )
    octave[0] = image
    oi = 1

    r2_map = r2_maps[idx]
    rad_max = np.max(r2_map)

    h, w = image.shape

    cx0 = (w - 1) / 2.0
    cy0 = (h - 1) / 2.0

    # for kernel in kernels[idx].values():
    #     # xi=0 => all r2 entries are the same kernel, so take any one
    #     ker = next(iter(kernel.values()))
    #     tmp = ndi.convolve1d(image, ker, axis=1, mode='reflect')
    #     blurred = ndi.convolve1d(tmp, ker, axis=0, mode='reflect')

    #     octave[oi] = blurred
    #     oi += 1
    #     image = blurred
    for kernel in tqdm(kernels[idx].values()):
        new_image = np.zeros_like(image)
        blurred = np.zeros_like(image)

        for i in range(h):
            for j in range(w):
                t = cx0 - i
                row = image[i]
                r2 = (r2_map[i,j] / rad_max)
                ker = kernel[r2]
                rad_ker = len(ker) // 2

                l_row = j
                r_row = w - j - 1

                left = min(l_row, rad_ker)
                right = min(r_row, rad_ker+1)
                #print("Hor ", j)
                kslice = ker[rad_ker - left: rad_ker + right]
                convolved = np.dot(row[j - left: j + right], kslice) / np.sum(kslice)
                new_image[i,j] = convolved
        for k in range(h):
            for m in range(w):
                t = cy0 - m
                row = new_image[:, m]
                r2 = (r2_map[k,m] / rad_max)
                ker = kernel[r2]
                rad_ker = len(ker) // 2

                l_row = k
                r_row = h - k - 1

                left = min(l_row, rad_ker)
                right = min(r_row, rad_ker+1)

                #print("vert ", k)

                kslice = ker[rad_ker - left: rad_ker + right]
                convolved = np.dot(row[k - left: k + right], kslice) / np.sum(kslice)

                blurred[k,m] = convolved

        #print('sum', np.sum(np.abs(blurred)))
        octave[oi] = blurred
        oi += 1
        image = blurred
    
    return octave

desc = sift.SIFT()
dog = desc._create_scalespace(image)

desc_rd = sift.SIFT()
desc_rd._create_1d_gaussians(image.shape, xi = 0)
dog_rd = desc_rd._create_rd_scalespace(image)

# upscaled = rescale(
#     image,
#     2,
#     order=1,
#     mode='reflect',
#     anti_aliasing=False,
#     preserve_range=True
# ).astype(np.float64)

# upscaled = ndi.gaussian_filter(
#     upscaled,
#     sigma=2 * math.sqrt(sigma_min**2 - sigma_in**2),
#     mode='reflect'
# )

# upscaled = dog[0][:,:,0]

# print(upscaled.shape)

# _, kernels, r2_maps, s = _create_1d_gaussians(upscaled.shape, 0)

# #ker2 = kernels[0][s[0, 0]][0]

# octave = _apply_kernels(kernels=kernels, r2_maps=r2_maps, image=upscaled, idx=0, granularity=1)
# img = octave[4].astype(np.uint8)


# _, kernels, r2_maps, s = _create_1d_gaussians(upscaled.shape, 0.5)

# #ker2 = kernels[0][s[0, 0]][0]

# octave2 = _apply_kernels(kernels=kernels, r2_maps=r2_maps, image=upscaled, idx=0, granularity=1)
# img2 = octave2[4].astype(np.uint8)

# print(octave.shape)



# for elt in kernels:
#     for ker in kernels[elt]:
#         for k in kernels[elt][ker]:
#             print(np.sum(np.abs(ker2 - kernels[elt][ker][k])))


# desc = sift.SIFT()
# dog = desc._create_scalespace(image)

print(dog[0].shape)

# print("base:", np.mean(np.abs(dog[0][:, :, 0] - upscaled)))
# print("s1  :", np.mean(np.abs(dog[0][:, :, 1] - octave[1])))
# print("s2  :", np.mean(np.abs(dog[0][:, :, 2] - octave[2])))
# print("s3  :", np.mean(np.abs(dog[0][:, :, 3] - octave[3])))
# print("s4  :", np.mean(np.abs(dog[0][:, :, 4] - octave[4])))

std = dog[0][:, :, 4].astype(np.uint8)
diff = dog[0][:, :, 4].astype(np.float64) - dog_rd[0][:,:,4]
print(diff.min(), diff.max(), np.mean(np.abs(diff)))
vis = cv2.normalize(np.abs(diff), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow("diff", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

#desc._create_1d_gaussians(image.shape, -0.5)

exit()