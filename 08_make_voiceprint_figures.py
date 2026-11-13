import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import cosine_similarity

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

# validated diverging pair: blue <-> red, neutral gray midpoint
div_cmap = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a08a", "#e34948", "#8a1f1f"]
)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
seq_cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)

speakers = [1, 2, 3, 4, 5]
F = np.load("./fp_F.npy")   # (5, 67) fingerprints (normal-range residual centroids)
C = np.load("./fp_C.npy")   # (5, 67) raw centroids
feat_cols = open("./fp_feat_cols.txt").read().splitlines()

# ============================================================
# Figure A: Discriminative Residual Voiceprints by Speaker
# ============================================================
groups = [
    ("MFCC (mean/std)", 0, 26), ("ΔMFCC", 26, 39), ("ΔΔMFCC", 39, 52),
    ("F0", 52, 54), ("Spectral", 54, 62), ("ZCR/RMS", 62, 66), ("Dur.", 66, 67),
]

fig, ax = plt.subplots(figsize=(15.5, 3.6))
vmax = np.max(np.abs(F))
im = ax.imshow(F, aspect="auto", cmap=div_cmap, vmin=-vmax, vmax=vmax)

ax.set_yticks(range(5))
ax.set_yticklabels([f"Speaker {s}" for s in speakers], fontsize=10)
ax.set_xticks([])
ax.set_xlabel("Acoustic feature index (67-dim residual vector, grouped)", fontsize=10)

# group separators + labels
for name, start, end in groups:
    ax.axvline(start - 0.5, color=SURFACE, linewidth=1.4, zorder=3)
    mid = (start + end - 1) / 2
    ax.text(mid, -0.85, name, ha="center", va="bottom", fontsize=8.7, color=INK2, rotation=0)
ax.axvline(66.5, color=SURFACE, linewidth=1.4, zorder=3)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
cbar.set_label("Fingerprint value\n(std. devs from the group's normal range)", fontsize=8.7, color=INK2)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_visible(False)

ax.set_title("Discriminative Residual Voiceprints by Speaker", fontsize=13.5, color=INK, pad=26)
fig.tight_layout()
fig.savefig("./fig_voiceprints_by_speaker.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig_voiceprints_by_speaker.png")

# ============================================================
# Figure B: Cosine Similarity Matrix — Raw vs Residual
# ============================================================
sim_raw = cosine_similarity(C)
sim_res = cosine_similarity(F)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

def plot_sim(ax, M, title):
    im = ax.imshow(M, cmap=div_cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"S{s}" for s in speakers])
    ax.set_yticklabels([f"S{s}" for s in speakers])
    ax.set_title(title, fontsize=11.5, color=INK, pad=10)
    for i in range(5):
        for j in range(5):
            val = M[i, j]
            color = "#ffffff" if abs(val) > 0.42 else INK
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return im

im1 = plot_sim(axes[0], sim_raw, "(a) Raw speaker centroids")
im2 = plot_sim(axes[1], sim_res, "(b) Residual fingerprints (normal-range)")

fig.suptitle("Cosine Similarity Matrix: Raw Features vs. Residual Features", fontsize=13.5, color=INK, y=1.02)
cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.62])
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.set_label("Cosine similarity", fontsize=9.5, color=INK2)
cbar.ax.tick_params(labelsize=8.5)
cbar.outline.set_visible(False)

off_raw = sim_raw[~np.eye(5, dtype=bool)].mean()
off_res = sim_res[~np.eye(5, dtype=bool)].mean()
fig.text(0.5, -0.03, f"Mean off-diagonal similarity — raw: {off_raw:.2f}   |   residual: {off_res:.2f}",
         ha="center", fontsize=9.5, color=INK2, style="italic")

fig.tight_layout()
fig.savefig("./fig_cosine_similarity_matrix.png", dpi=200, bbox_inches="tight")
print("saved fig_cosine_similarity_matrix.png")
print("mean off-diag raw:", round(off_raw,4), "residual:", round(off_res,4))
print("sim_raw:\n", np.round(sim_raw,3))
print("sim_res:\n", np.round(sim_res,3))
