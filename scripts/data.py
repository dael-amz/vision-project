import numpy as np
import matplotlib.pyplot as plt

A = 1 / 1.333

results = np.zeros((5, 10, 18, 10))

amps = np.linspace(0.0, 0.002, 10)
angles = np.linspace(0.01, 1, 10)

xis = angles * A * (1- A**2) / (2 * (1 + angles))
print(xis)

for i in range(5):
    for ang in range(len(angles)):
        arr = np.loadtxt(f"flow-{i}-{angles[ang]}-results")
        results[i,ang] = arr
# -------------------------
srd_s = np.stack((results[:, :, 13, :], results[:, :, 12, :]), axis=3)
num_srd_s = np.min(srd_s, axis=3)
srd_s_true = num_srd_s * results[:, :, 0, :]
srd_m_true = srd_s_true * results[:, :, 1, :]
srd_matches = srd_m_true * results[:, :, 2, :]


# -------------------------
# sRD-SIFT + Flow
# -------------------------
srd_flow_s = np.stack((results[:, :, 14, :], results[:, :, 12, :]), axis=3)
num_srd_flow_s = np.min(srd_flow_s, axis=3)
srd_flow_s_true = num_srd_flow_s * results[:, :, 3, :]
srd_flow_m_true = srd_flow_s_true * results[:, :, 4, :]
srd_flow_matches = srd_flow_m_true * results[:, :, 5, :]

# -------------------------
# SIFT
# -------------------------
sift_s = np.stack((results[:, :, 15, :], results[:, :, 12, :]), axis=3)
num_sift_s = np.min(sift_s, axis=3)
sift_s_true = num_sift_s * results[:, :, 6, :]
sift_m_true = sift_s_true * results[:, :, 7, :]
sift_matches = sift_m_true * results[:, :, 8, :]

# -------------------------
# SIFT + Flow
# -------------------------
# NOTE:
# This assumes indices 14 and 15 are NOT already used for keypoint counts
# of sRD+Flow in your file format. If your tensor really has only 18 slots,
# then you may not actually have two extra keypoint-count channels left for
# SIFT+Flow. In that case this block needs to be adjusted to your real layout.
sift_flow_s = np.stack((results[:, :, 16, :], results[:, :, 12, :]), axis=3)
num_sift_flow_s = np.min(sift_flow_s, axis=3)
sift_flow_s_true = num_sift_flow_s * results[:, :, 9, :]
sift_flow_m_true = sift_flow_s_true * results[:, :, 10, :]
sift_flow_matches = sift_flow_m_true * results[:, :, 11, :]

avg_srd_m_true = np.mean(srd_m_true, axis=0)
avg_srd_flow_m_true = np.mean(srd_flow_m_true, axis=0)
avg_sift_m_true = np.mean(sift_m_true, axis=0)
avg_sift_flow_m_true = np.mean(sift_flow_m_true, axis=0)

#import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Average over runs
# -----------------------------
avg_srd_m_true = np.mean(srd_m_true, axis=0)            # shape: (n_angles, n_amps)
avg_srd_flow_m_true = np.mean(srd_flow_m_true, axis=0)
avg_sift_m_true = np.mean(sift_m_true, axis=0)
avg_sift_flow_m_true = np.mean(sift_flow_m_true, axis=0)

# -----------------------------
# Shared color scale
# -----------------------------
all_data = [
    avg_srd_m_true,
    avg_srd_flow_m_true,
    avg_sift_m_true,
    avg_sift_flow_m_true
]
vmin = min(d.min() for d in all_data)
vmax = max(d.max() for d in all_data)

# -----------------------------
# Figure
# -----------------------------
fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

titles = [
    'sRD-SIFT',
    'sRD-SIFT + Flow',
    'SIFT',
    'SIFT + Flow'
]

datasets = [
    avg_srd_m_true,
    avg_srd_flow_m_true,
    avg_sift_m_true,
    avg_sift_flow_m_true
]

axes = ax.ravel()
ims = []

for k, (a, data, title) in enumerate(zip(axes, datasets, titles)):
    im = a.imshow(
        data,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        aspect='equal'
    )
    ims.append(im)

    a.set_title(title, fontsize=11)
    a.set_xlabel('Amplitude', fontsize=9)
    a.set_ylabel('Angle', fontsize=9)

    # Tick positions at cell centers
    a.set_xticks(np.arange(len(amps)))
    a.set_yticks(np.arange(len(angles)))

    # Fewer tick labels so it stays clean
    xtick_idx = np.arange(0, len(amps), 2)
    ytick_idx = np.arange(0, len(angles), 2)

    a.set_xticks(xtick_idx)
    a.set_yticks(ytick_idx)
    a.set_xticklabels([f'{amps[j]*1000:.1f}' for j in xtick_idx])
    a.set_xlabel('Amplitude (×10⁻³)', fontsize=9)
    a.set_yticklabels([f'{angles[i]:.2f}' for i in ytick_idx], fontsize=8)

    # Minor gridlines to show squares clearly
    a.set_xticks(np.arange(-0.5, len(amps), 1), minor=True)
    a.set_yticks(np.arange(-0.5, len(angles), 1), minor=True)
    a.grid(which='minor', color='white', linestyle='-', linewidth=0.8)
    a.tick_params(which='minor', bottom=False, left=False)

    # Put values inside cells using matrix indices
    threshold = 0.55 * (vmin + vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = 'white' if val < threshold else 'black'
            a.text(
                j, i, f'{val:.1f}',
                ha='center',
                va='center',
                fontsize=6.5,
                color=text_color
            )

# Shared colorbar
cbar = fig.colorbar(ims[0], ax=ax, shrink=0.92)
cbar.set_label(r'Average $M_{\mathrm{true}}$', fontsize=10)
cbar.ax.tick_params(labelsize=8)

plt.show()
# from matplotlib.lines import Line2D
# import numpy as np
# import matplotlib.pyplot as plt

# # --- averages over trials ---
# srd_rec   = np.mean(results[:, :, 1, :], axis=0)   # (angles, amps)
# srd_prec  = np.mean(results[:, :, 2, :], axis=0)

# srd_f_rec  = np.mean(results[:, :, 4, :], axis=0)
# srd_f_prec = np.mean(results[:, :, 5, :], axis=0)

# sift_rec   = np.mean(results[:, :, 7, :], axis=0)
# sift_prec  = np.mean(results[:, :, 8, :], axis=0)

# sift_f_rec  = np.mean(results[:, :, 10, :], axis=0)
# sift_f_prec = np.mean(results[:, :, 11, :], axis=0)

# # --- top 3 amplitudes ---
# from matplotlib.lines import Line2D
# import numpy as np
# import matplotlib.pyplot as plt

# # --- averages over trials ---
# srd_rec   = np.mean(results[:, :, 1, :], axis=0)   # shape: (angles, amps)
# srd_prec  = np.mean(results[:, :, 2, :], axis=0)

# srd_f_rec  = np.mean(results[:, :, 4, :], axis=0)
# srd_f_prec = np.mean(results[:, :, 5, :], axis=0)

# sift_rec   = np.mean(results[:, :, 7, :], axis=0)
# sift_prec  = np.mean(results[:, :, 8, :], axis=0)

# sift_f_rec  = np.mean(results[:, :, 10, :], axis=0)
# sift_f_prec = np.mean(results[:, :, 11, :], axis=0)

# # --- choose amplitudes: lowest non-zero, middle, highest ---
# amp_idx = np.array([1, len(amps) // 2, len(amps) - 1])

# fig, ax = plt.subplots(figsize=(5.8, 4.8))

# markers = ['o', 's', '^']
# amp_legend = []

# for k, J in enumerate(amp_idx):
#     marker = markers[k]

#     # --- sRD-SIFT ---
#     x = srd_prec[:, J]
#     y = srd_rec[:, J]
#     o = np.argsort(x)
#     ax.plot(
#         x[o], y[o],
#         color='#1f77b4',
#         linestyle='-',
#         marker=marker,
#         markersize=5,
#         linewidth=1.8
#     )

#     # --- sRD-SIFT + Flow ---
#     x = srd_f_prec[:, J]
#     y = srd_f_rec[:, J]
#     o = np.argsort(x)
#     ax.plot(
#         x[o], y[o],
#         color='#17becf',
#         linestyle='--',
#         marker=marker,
#         markersize=5,
#         linewidth=1.8
#     )

#     # --- SIFT ---
#     x = sift_prec[:, J]
#     y = sift_rec[:, J]
#     o = np.argsort(x)
#     ax.plot(
#         x[o], y[o],
#         color='#d62728',
#         linestyle='-',
#         marker=marker,
#         markersize=5,
#         linewidth=1.8
#     )

#     # --- SIFT + Flow ---
#     x = sift_f_prec[:, J]
#     y = sift_f_rec[:, J]
#     o = np.argsort(x)
#     ax.plot(
#         x[o], y[o],
#         color='#ff9896',
#         linestyle='--',
#         marker=marker,
#         markersize=5,
#         linewidth=1.8
#     )

#     # --- legend entry for amplitude ---
#     amp_legend.append(
#         Line2D(
#             [0], [0],
#             marker=marker,
#             color='black',
#             linestyle='None',
#             markersize=6,
#             label=f'{amps[J]:.4f}'
#         )
#     )

# # --- legends ---
# leg1 = ax.legend(handles=amp_legend, title='Amplitude', fontsize=7, loc='lower left')
# ax.add_artist(leg1)

# method_legend = [
#     Line2D([0], [0], color='#1f77b4', linestyle='-',  linewidth=2, label='sRD-SIFT'),
#     Line2D([0], [0], color='#17becf', linestyle='--', linewidth=2, label='sRD-SIFT + Flow'),
#     Line2D([0], [0], color='#d62728', linestyle='-',  linewidth=2, label='SIFT'),
#     Line2D([0], [0], color='#ff9896', linestyle='--', linewidth=2, label='SIFT + Flow'),
# ]
# ax.legend(handles=method_legend, fontsize=8, loc='lower right')

# # --- styling ---
# ax.set_xlabel('Precision', fontsize=10)
# ax.set_ylabel('Recall', fontsize=10)
# ax.set_title('Recall vs Precision', fontsize=11)

# ax.tick_params(labelsize=8)
# ax.spines[['top', 'right']].set_visible(False)

# plt.tight_layout()
# plt.show()

# from matplotlib.lines import Line2D

# import numpy as np
# import matplotlib.pyplot as plt

# plt.style.use('seaborn-v0_8-whitegrid')

# A = 1 / 1.333

# results = np.zeros((5, 10, 18, 10))

# amps = np.linspace(0.0, 0.002, 10)
# angles = np.linspace(0.01, 1, 10)

# for i in range(5):
#     for ang in range(len(angles)):
#         arr = np.loadtxt(f"flow-{i}-{angles[ang]}-results")
#         results[i, ang] = arr

# # fixed angle index, same as in your amplitude plots
# ang_idx = 1

# # --- average over trials ---
# srd_rep  = np.mean(results[:, ang_idx, 0, :], axis=0)
# srd_rec  = np.mean(results[:, ang_idx, 1, :], axis=0)
# srd_prec = np.mean(results[:, ang_idx, 2, :], axis=0)

# srd_flow_rep  = np.mean(results[:, ang_idx, 3, :], axis=0)
# srd_flow_rec  = np.mean(results[:, ang_idx, 4, :], axis=0)
# srd_flow_prec = np.mean(results[:, ang_idx, 5, :], axis=0)

# sift_rep  = np.mean(results[:, ang_idx, 6, :], axis=0)
# sift_rec  = np.mean(results[:, ang_idx, 7, :], axis=0)
# sift_prec = np.mean(results[:, ang_idx, 8, :], axis=0)

# sift_flow_rep  = np.mean(results[:, ang_idx, 9, :], axis=0)
# sift_flow_rec  = np.mean(results[:, ang_idx, 10, :], axis=0)
# sift_flow_prec = np.mean(results[:, ang_idx, 11, :], axis=0)

# plt.style.use('seaborn-v0_8-whitegrid')

# fig = plt.figure(figsize=(6, 6.5))
# gs = fig.add_gridspec(2, 2)

# ax0 = fig.add_subplot(gs[0, 0])
# ax1 = fig.add_subplot(gs[0, 1])
# ax2 = fig.add_subplot(gs[1, :])

# # --- Colors (match previous style) ---
# colors = {
#     'srd': '#1f77b4',        # blue
#     'srd_flow': '#17becf',   # teal
#     'sift': '#d62728',       # red
#     'sift_flow': '#ff9896'   # light red
# }

# lw = 2
# ms = 4

# # --- Repeatability ---
# ax0.plot(amps, srd_rep,
#          color=colors['srd'], linewidth=lw, linestyle='-',
#          marker='o', markersize=ms, label='sRD-SIFT')

# ax0.plot(amps, srd_flow_rep,
#          color=colors['srd_flow'], linewidth=lw, linestyle='--',
#          marker='^', markersize=ms, label='sRD-SIFT + Flow')

# ax0.plot(amps, sift_rep,
#          color=colors['sift'], linewidth=lw, linestyle='-',
#          marker='s', markersize=ms, label='SIFT')

# ax0.plot(amps, sift_flow_rep,
#          color=colors['sift_flow'], linewidth=lw, linestyle='--',
#          marker='D', markersize=ms, label='SIFT + Flow')

# ax0.set_title('Repeatability', fontsize=10)
# ax0.set_xlabel('Amplitude', fontsize=9)
# ax0.set_ylabel('Score', fontsize=9)

# # --- Recall ---
# ax1.plot(amps, srd_rec,
#          color=colors['srd'], linewidth=lw, linestyle='-',
#          marker='o', markersize=ms)

# ax1.plot(amps, srd_flow_rec,
#          color=colors['srd_flow'], linewidth=lw, linestyle='--',
#          marker='^', markersize=ms)

# ax1.plot(amps, sift_rec,
#          color=colors['sift'], linewidth=lw, linestyle='-',
#          marker='s', markersize=ms)

# ax1.plot(amps, sift_flow_rec,
#          color=colors['sift_flow'], linewidth=lw, linestyle='--',
#          marker='D', markersize=ms)

# ax1.set_title('Recall', fontsize=10)
# ax1.set_xlabel('Amplitude', fontsize=9)

# # --- Precision ---
# ax2.plot(amps, srd_prec,
#          color=colors['srd'], linewidth=lw, linestyle='-',
#          marker='o', markersize=ms)

# ax2.plot(amps, srd_flow_prec,
#          color=colors['srd_flow'], linewidth=lw, linestyle='--',
#          marker='^', markersize=ms)

# ax2.plot(amps, sift_prec,
#          color=colors['sift'], linewidth=lw, linestyle='-',
#          marker='s', markersize=ms)

# ax2.plot(amps, sift_flow_prec,
#          color=colors['sift_flow'], linewidth=lw, linestyle='--',
#          marker='D', markersize=ms)

# ax2.set_title('Precision', fontsize=10)
# ax2.set_xlabel('Amplitude', fontsize=9)
# ax2.set_ylabel('Score', fontsize=9)

# # --- (a), (b), (c) labels ---
# labels = ['(a)', '(b)', '(c)']
# for i, a in enumerate([ax0, ax1, ax2]):
#     a.text(0.98, 0.98, labels[i],
#            transform=a.transAxes,
#            fontsize=11,
#            fontweight='bold',
#            ha='right',
#            va='top')

# # --- styling ---
# for a in [ax0, ax1, ax2]:
#     a.tick_params(labelsize=8)
#     a.spines[['top', 'right']].set_visible(False)

# # --- single legend ---
# ax0.legend(fontsize=7, frameon=True)

# plt.tight_layout()
# plt.show()

# --- average over runs (axis=0) ---
# srd_flow_recall = np.mean(results[:, :, 4, :], axis=0)   # (angles, amps)
# srd_flow_precision = np.mean(results[:, :, 5, :], axis=0)

# sift_flow_recall = np.mean(results[:, :, 10, :], axis=0)
# sift_flow_precision = np.mean(results[:, :, 11, :], axis=0)

# fig, ax = plt.subplots(figsize=(5.5, 4.5))

# markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'x']

# legend_elements = []

# for I in range(0, len(angles), 3):
#     marker = markers[I % len(markers)]

#     # --- sRD-SIFT + Flow ---
#     order_srd = np.argsort(srd_flow_precision[I, :])
#     x_srd = srd_flow_precision[I, :][order_srd]
#     y_srd = srd_flow_recall[I, :][order_srd]

#     ax.plot(
#         x_srd, y_srd,
#         color='#1f77b4',
#         linestyle='-',
#         marker=marker,
#         markersize=4,
#         linewidth=1.8,
#         alpha=0.9
#     )

#     # --- SIFT + Flow ---
#     order_sift = np.argsort(sift_flow_precision[I, :])
#     x_sift = sift_flow_precision[I, :][order_sift]
#     y_sift = sift_flow_recall[I, :][order_sift]

#     ax.plot(
#         x_sift, y_sift,
#         color='#7f7f7f',
#         linestyle='--',
#         marker=marker,
#         markersize=4,
#         linewidth=1.8,
#         alpha=0.9
#     )

#     # --- legend entry (marker = angle) ---
#     legend_elements.append(
#         Line2D([0], [0],
#                marker=marker,
#                color='black',
#                linestyle='None',
#                label=f'{angles[I]:.2f}')
#     )

# # --- Labels & style ---
# ax.set_xlabel('Precision', fontsize=10)
# ax.set_ylabel('Recall', fontsize=10)
# ax.set_title('Recall vs Precision (Flow Filtering)', fontsize=11)

# ax.tick_params(labelsize=8)
# ax.spines[['top', 'right']].set_visible(False)

# # --- Legend for angles (markers) ---
# leg1 = ax.legend(handles=legend_elements, title='Angle', fontsize=7, loc='lower left')
# ax.add_artist(leg1)

# # --- Legend for method ---
# method_legend = [
#     Line2D([0], [0], color='#1f77b4', linestyle='-', label='sRD-SIFT + Flow'),
#     Line2D([0], [0], color='#7f7f7f', linestyle='--', label='SIFT + Flow')
# ]
# ax.legend(handles=method_legend, fontsize=8, loc='lower right')

# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(ncols=3, figsize=(11, 4))

# avg_srd_repeatability_no_wave = np.mean(results[:, 1, 0, :], axis = 0)
# avg_srd_flow_repeatability_no_wave = np.mean(results[:, 1, 3, :], axis = 0)
# avg_sift_repeatability_no_wave = np.mean(results[:, 1, 6, :], axis = 0)
# avg_sift_flow_repeatability_no_wave = np.mean(results[:, 1, 9, :], axis = 0)

# plt.style.use('seaborn-v0_8-whitegrid')

# fig, ax = plt.subplots(figsize=(6, 4.2))

# colors = {
#     'srd': '#1f77b4',
#     'srd_flow': '#17becf',
#     'sift': '#d62728',
#     'sift_flow': '#ff9896'
# }

# lw = 2
# ms = 4

# ax.plot(
#     amps, avg_srd_repeatability_no_wave,
#     color=colors['srd'],
#     linewidth=lw,
#     linestyle='-',
#     marker='o',
#     markersize=ms,
#     label='sRD-SIFT'
# )

# ax.plot(
#     amps, avg_srd_flow_repeatability_no_wave,
#     color=colors['srd_flow'],
#     linewidth=lw,
#     linestyle='--',
#     marker='^',
#     markersize=ms,
#     label='sRD-SIFT + Flow'
# )

# ax.plot(
#     amps, avg_sift_repeatability_no_wave,
#     color=colors['sift'],
#     linewidth=lw,
#     linestyle='-',
#     marker='s',
#     markersize=ms,
#     label='SIFT'
# )

# ax.plot(
#     amps, avg_sift_flow_repeatability_no_wave,
#     color=colors['sift_flow'],
#     linewidth=lw,
#     linestyle='--',
#     marker='D',
#     markersize=ms,
#     label='SIFT + Flow'
# )

# ax.set_xlabel('Amplitude', fontsize=10)
# ax.set_ylabel('Repeatability', fontsize=10)
# ax.set_title('Repeatability vs Wave Amplitude', fontsize=11)

# ax.tick_params(labelsize=8)
# ax.spines[['top', 'right']].set_visible(False)
# ax.legend(fontsize=8, frameon=True)

# plt.tight_layout()
# plt.show()

# avg_srd_repeatability_no_wave = np.mean(results[:, :, 0, 0], axis = 0)
# avg_sift_repeatability_no_wave = np.mean(results[:, :, 6, 0], axis = 0)


# #for ang in range(len(angles)):
# xis = angles# * A * (1-A**2) / (2 * (1 + angles))
# # ax[0].plot(xis, avg_srd_repeatability_no_wave, color = 'blue', label = 'sRD-SIFT')
# # ax[0].plot(xis, avg_sift_repeatability_no_wave, color = 'red', label = 'SIFT')

# srd_recall_nw = np.mean(results[:, :, 1, 0], axis = 0)
# sift_recall_nw = np.mean(results[:, :, 7, 0], axis = 0)

# # ax[1].plot(xis, srd_recall_nw, color = 'blue')
# # ax[1].plot(xis, sift_recall_nw, color = 'red')

# srd_precision_new = np.mean(results[:, :, 2, 0], axis = 0)
# sift_precision_new = np.mean(results[:, :, 8, 0], axis = 0)

# # ax[2].plot(xis, srd_precision_new, color = 'blue')
# # ax[2].plot(xis, sift_precision_new, color = 'red')


# plt.style.use('seaborn-v0_8-whitegrid')

# fig = plt.figure(figsize=(6, 6))  # narrower for 2-column papers
# gs = fig.add_gridspec(2, 2)

# ax0 = fig.add_subplot(gs[0, 0])   # top-left
# ax1 = fig.add_subplot(gs[0, 1])   # top-right
# ax2 = fig.add_subplot(gs[1, :])   # bottom spanning both columns

# colors = {'srd': '#1f77b4', 'sift': '#d62728'}
# lw = 2
# ms = 4

# # --- Repeatability ---
# ax0.plot(xis, avg_srd_repeatability_no_wave,
#          color=colors['srd'], linewidth=lw, marker='o', markersize=ms, label='sRD-SIFT')
# ax0.plot(xis, avg_sift_repeatability_no_wave,
#          color=colors['sift'], linewidth=lw, linestyle='--', marker='s', markersize=ms, label='SIFT')
# ax0.set_title('Repeatability', fontsize=10)
# ax0.set_xlabel(r'$\lambda$', fontsize=9)
# ax0.set_ylabel('Score', fontsize=9)

# # --- Recall ---
# ax1.plot(xis, srd_recall_nw,
#          color=colors['srd'], linewidth=lw, marker='o', markersize=ms)
# ax1.plot(xis, sift_recall_nw,
#          color=colors['sift'], linewidth=lw, linestyle='--', marker='s', markersize=ms)
# ax1.set_title('Recall', fontsize=10)
# ax1.set_xlabel(r'$\lambda$', fontsize=9)

# # --- Precision (bottom, full width) ---
# ax2.plot(xis, srd_precision_new,
#          color=colors['srd'], linewidth=lw, marker='o', markersize=ms)
# ax2.plot(xis, sift_precision_new,
#          color=colors['sift'], linewidth=lw, linestyle='--', marker='s', markersize=ms)
# ax2.set_title('Precision', fontsize=10)
# ax2.set_xlabel(r'$\lambda$', fontsize=9)
# ax2.set_ylabel('Score', fontsize=9)

# # --- Legend (only once) ---
# ax0.legend(fontsize=8, frameon=True)

# # --- Clean style ---
# for a in [ax0, ax1, ax2]:
#     a.tick_params(labelsize=8)
#     a.spines[['top', 'right']].set_visible(False)

# # --- subplot labels (a), (b), (c) ---
# labels = ['(a)', '(b)', '(c)']
# axes = [ax0, ax1, ax2]

# for i, a in enumerate(axes):
#     a.text(-0.08, 1.05, labels[i], transform=a.transAxes,
#        fontsize=11, fontweight='bold')

# plt.tight_layout()
# plt.show()


