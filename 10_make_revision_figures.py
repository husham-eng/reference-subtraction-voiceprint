"""
Figures for the revised (camera-ready) paper:
  fig_conceptual_model.png   Fig. 1 - research design, reductions, and findings
  fig_confusion_revised.png  Fig. 2 - confusion matrices (nearest centroid vs normal-range
                                      reference subtraction = diagonal Mahalanobis)
  fig_session_shift.png      Fig. 3 - train-to-test shift per speaker (separate sessions)
Run after 02_run_experiments.py and 09_reviewer_checks.py.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
DIV = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a08a", "#e34948", "#8a1f1f"]
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
R = json.load(open("reviewer_checks.json"))
E = json.load(open("experiments_results.json"))

# ---------------- Fig. 1: conceptual model ----------------
fig, ax = plt.subplots(figsize=(3.5, 4.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 12.3); ax.axis("off")

def box(x, y, w, h, txt, fc, tc="white", fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", lw=0, fc=fc))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", color=tc, fontsize=fs,
            linespacing=1.25, fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))

box(0.1, 10.35, 9.8, 1.85, "DESIGN\n5 speakers (3 men, 2 women), one Arabic word,\none phone per speaker; training and test sessions\non different days; 67 features (52 cepstral +\n2 pitch + 8 spectral + 5 energy/duration)",
    "#52514e", fs=5.6)
arrow(5, 10.3, 5, 10.0)
ax.text(2.45, 9.7, "Variant", ha="center", fontsize=6.2, color=INK, fontweight="bold")
ax.text(7.6, 9.7, "Reduces to (test set)", ha="center", fontsize=6.2, color=INK, fontweight="bold")
rows = [
    ("Mean only + Euclidean", "Nearest centroid\nidentical on 180/180", CAT[0]),
    ("Normal range + diagonal\nMahalanobis (best macro-F1)", "Plain diagonal Mahalanobis\nidentical on 180/180", CAT[2]),
    ("Normal range + Euclidean", "Centroid distance scaled by\nother speakers' spread (F1 0.805)", CAT[3]),
    ("Any reference + cosine", "No reduction; not better\n(macro-F1 0.762-0.764)", CAT[1]),
]
y = 8.2
for left, right, col in rows:
    box(0.1, y, 4.7, 1.25, left, col, tc=INK if col in (CAT[3],) else "white", fs=5.9)
    arrow(4.85, y + 0.62, 5.25, y + 0.62)
    box(5.3, y, 4.6, 1.25, right, "#f0efec", tc=INK, fs=5.7)
    y -= 1.45
arrow(5, y + 1.4, 5, y + 0.95)
box(0.1, 0.15, 9.8, y + 0.75,
    "FINDINGS\n"
    "Accuracy 0.850 for the three best rules\n"
    "Macro-F1 gain +0.010 (95% interval -0.054 to +0.073)\n"
    "Paired test: 22 vs 22 discordant, exact p = 1.00\n"
    "Same-session cross-validation 0.95-0.99, but\n"
    "separate-session test 0.76-0.84: session and\n"
    "device effects dominate; reference subtraction\n"
    "adds nothing beyond the scoring metric",
    "#0d366b", fs=5.6)
fig.tight_layout(pad=0.1)
fig.savefig("fig_conceptual_model.png", dpi=300)
plt.close(fig)

# ---------------- Fig. 2: confusion matrices ----------------
seq = LinearSegmentedColormap.from_list("seq", SEQ)
spk = E["speakers"]
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1))
for ax, key, title in [(axes[0], "Baseline-Centroid-Euclid", "(a) Nearest centroid"),
                       (axes[1], "RefSub-NormalRange-MahalanobisDiag", "(b) Normal-range reference\n= diagonal Mahalanobis")]:
    cm = np.array(E["confusion_matrices"][key])
    ax.imshow(cm, cmap=seq, vmin=0, vmax=48)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > 24 else INK)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"S{s}" for s in spk], fontsize=5.5); ax.set_yticklabels([f"S{s}" for s in spk], fontsize=5.5)
    ax.set_xlabel("Predicted speaker", fontsize=5.8); ax.set_title(title, fontsize=6.2, pad=3)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_ylabel("True speaker", fontsize=5.8)
fig.tight_layout(pad=0.2)
fig.savefig("fig_confusion_revised.png", dpi=300)
plt.close(fig)

# ---------------- Fig. 3: session shift ----------------
df = pd.read_csv("features.csv")
feats = [("duration", "Duration"), ("rms_mean", "Energy (root mean square)"), ("f0_mean", "Pitch (mean)"),
         ("mfcc1_mean", "Cepstral coeff. 1 (log-energy)"), ("centroid_mean", "Spectral centroid")]
M = np.zeros((len(feats), 5))
for i, (f, _) in enumerate(feats):
    g = df.groupby(["speaker", "split"])[f].mean().unstack()
    sd = df[df.split == "train"].groupby("speaker")[f].std()
    M[i] = ((g["test"] - g["train"]) / sd).values
div = LinearSegmentedColormap.from_list("div", DIV)
fig, ax = plt.subplots(figsize=(3.5, 1.75))
im = ax.imshow(M, cmap=div, vmin=-4, vmax=4, aspect="auto")
for i in range(M.shape[0]):
    for j in range(5):
        ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=5.8,
                color="white" if abs(M[i, j]) > 2.2 else INK)
ax.set_xticks(range(5)); ax.set_xticklabels([f"S{s}" for s in spk], fontsize=5.6); ax.set_xlabel("Speaker", fontsize=5.8)
ax.set_yticks(range(len(feats))); ax.set_yticklabels([n for _, n in feats], fontsize=5.6)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.ax.tick_params(labelsize=5.2); cb.outline.set_visible(False)
cb.set_label("Test minus train mean\n(training standard deviations)", fontsize=5.2)
fig.tight_layout(pad=0.2)
fig.savefig("fig_session_shift.png", dpi=300)
print("saved 3 figures")
