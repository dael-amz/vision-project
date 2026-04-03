import numpy as np
import matplotlib.pyplot as plt

A = 1.0 / 1.33333


def divison(x, h, d):
    xi = d * A * (1-A**2) / (2* (h + d))
    r2 = x**2
    res = x / (1 + xi * r2)
    return res

def water(x, h, d):
    r2 = x**2
    n = 1 / (h + d)
    div = 1 / np.sqrt(1+(1-A**2) * r2)
    res = h * x + d * A * div * x
    res *= n
    return res

granularity = 1000
rng = np.linspace(-1, 1, granularity)

H = np.arange(0.1, 100, 0.1)
D = np.arange(0.1, 100, 0.1)
angles = np.arange(0.1, 100, 0.1)


results = np.zeros((len(angles)))

for a in range(len(angles)):
    res = np.zeros(len(H))
    for h in range(len(H)):
        #for d in range(len(D)):
            res1 = (divison(rng, H[h], H[h] * a))
            res2 = water(rng, H[h], H[h] * a)
            diff = np.max(np.abs(res1-res2))
            res[h] = diff
    var = np.var(res)
    results[a] = var

print(np.var(results))

plt.plot(H, results)
plt.show()

# for h in range(len(H)):
#     for d in range(len(D)):
#         if abs(results[h, d] - 0.089) < 0.001:
#             results[h,d] = 100

# fig, ax = plt.subplots()


# cax = ax.matshow(results)
# fig.colorbar(cax, label='Max |f_d-g_r|')

# x_target = np.argmin(np.abs(D - 70))   # column index
# y_target = np.argmin(np.abs(H - 100))
# ax.plot([0, x_target], [0, y_target], color='red', linewidth=2)

# # mask = np.abs(results - 0.092) < 0.001
# # ys, xs = np.where(mask)
# # plt.plot(xs, ys, 'r.', markersize=2)

# # plt.plot([0, 1], [0, 1])

# ax.xaxis.set_label_position('top')   # move label
# ax.xaxis.tick_top() 

# ax.set_xlabel('Water Depth')
# ax.set_ylabel('Camera Height')

# ax.set_xticks(
#     ticks=np.linspace(0, len(D)-1, 6),
#     labels=np.round(np.linspace(D[0], D[-1], 6), 1)
# )

# ax.set_yticks(
#     ticks=np.linspace(0, len(H)-1, 6),
#     labels=np.round(np.linspace(H[0], H[-1], 6), 1)
# )
# plt.show()