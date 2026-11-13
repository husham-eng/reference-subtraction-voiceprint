import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

with open("./experiments_results.json") as f:
    R = json.load(f)

speakers = R["speakers"]
cms = R["confusion_matrices"]
summary = R["summary"]

# ---- palette (validated) ----
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # 5 speakers, fixed order
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]  # blue seq ramp

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

from matplotlib.colors import LinearSegmentedColormap
seq_cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)

def plot_confusion(name, ax, title):
    cm = np.array(cms[name])
    im = ax.imshow(cm, cmap=seq_cmap, vmin=0, vmax=cm.max())
    n = len(speakers)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"S{s}" for s in speakers])
    ax.set_yticklabels([f"S{s}" for s in speakers])
    ax.set_xlabel("Predicted speaker")
    ax.set_ylabel("True speaker")
    ax.set_title(title, color=INK, fontsize=11, pad=10)
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            color = "#ffffff" if val > thresh else INK
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plot_confusion("Baseline-Centroid-Euclid", axes[0], "(a) Baseline nearest-centroid\n(Euclidean, no reference subtraction)")
plot_confusion("RefSub-NormalRange-MahalanobisDiag", axes[1], "(b) Reference Subtraction\n(normal-range, Mahalanobis-diag)")
fig.suptitle("Confusion matrices on the held-out test set (n=180)", color=INK, fontsize=13, y=1.03)
fig.tight_layout()
fig.savefig("./fig_confusion_matrices.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig_confusion_matrices.png")

# ---- accuracy / macro-F1 comparison bar chart ----
order = [
    "Baseline-Centroid-Cosine", "Baseline-Centroid-Euclid",
    "RefSub-MeanOnly-Cosine", "RefSub-MeanOnly-Euclid",
    "RefSub-NormalRange-Cosine", "RefSub-NormalRange-Euclid",
    "RefSub-NormalRange-MahalanobisDiag",
]
labels_short = [
    "Baseline\n(cosine)", "Baseline\n(Euclid)",
    "RefSub mean-only\n(cosine)", "RefSub mean-only\n(Euclid)",
    "RefSub normal-range\n(cosine)", "RefSub normal-range\n(Euclid)",
    "RefSub normal-range\n(Mahalanobis)",
]
smap = {m["method"]: m for m in summary}
acc = [smap[m]["accuracy"] for m in order]
f1 = [smap[m]["macro_f1"] for m in order]

x = np.arange(len(order))
width = 0.36

fig, ax = plt.subplots(figsize=(12, 5.5))
b1 = ax.bar(x - width/2, acc, width, label="Accuracy", color=CAT[0])
b2 = ax.bar(x + width/2, f1, width, label="Macro F1", color=CAT[1])

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK)

ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=8.8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Test-set accuracy and macro-F1 across method variants (n=180)", fontsize=12, color=INK)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
fig.tight_layout()
fig.savefig("./fig_method_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig_method_comparison.png")

# ---- per-speaker recall comparison (baseline vs best refsub) ----
rep_base = R["per_method_classification_report"]["Baseline-Centroid-Euclid"]
rep_ref = R["per_method_classification_report"]["RefSub-NormalRange-MahalanobisDiag"]
recall_base = [rep_base[str(s)]["recall"] for s in speakers]
recall_ref = [rep_ref[str(s)]["recall"] for s in speakers]

x = np.arange(len(speakers))
width = 0.36
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - width/2, recall_base, width, label="Baseline (Euclid centroid)", color=CAT[0])
b2 = ax.bar(x + width/2, recall_ref, width, label="RefSub normal-range (Mahalanobis)", color=CAT[2])
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f"Speaker {s}" for s in speakers])
ax.set_ylim(0, 1.15)
ax.set_ylabel("Recall")
ax.set_title("Per-speaker recall: baseline vs. Reference Subtraction", fontsize=12, color=INK)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig("./fig_per_speaker_recall.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig_per_speaker_recall.png")
