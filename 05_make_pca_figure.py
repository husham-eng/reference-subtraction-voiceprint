# NOTE (revision): not used in the revised paper. Test residuals here use each sample's TRUE
# speaker reference, i.e. test labels, so the projection is not a valid discrimination diagnostic.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

Xtr = np.load("./pca_Xtr.npy"); Xte = np.load("./pca_Xte.npy")
Ztr = np.load("./pca_Ztr.npy"); Zte = np.load("./pca_Zte.npy")
ytr = np.load("./pca_ytr.npy"); yte = np.load("./pca_yte.npy")
speakers = sorted(np.unique(ytr).tolist())

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed categorical order, 5 speakers

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

def fit_transform(Xtr, Xte):
    pca = PCA(n_components=2, random_state=0)
    Ptr = pca.fit_transform(Xtr)
    Pte = pca.transform(Xte)
    var = pca.explained_variance_ratio_ * 100
    return Ptr, Pte, var

Ptr_raw, Pte_raw, var_raw = fit_transform(Xtr, Xte)
Ptr_res, Pte_res, var_res = fit_transform(Ztr, Zte)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

def plot_panel(ax, Ptr, Pte, var, title):
    for i, s in enumerate(speakers):
        m_tr = ytr == s
        m_te = yte == s
        ax.scatter(Ptr[m_tr, 0], Ptr[m_tr, 1], s=26, color=CAT[i], alpha=0.85,
                   marker="o", linewidths=0, label=f"Speaker {s}", zorder=3)
        ax.scatter(Pte[m_te, 0], Pte[m_te, 1], s=34, facecolors="none",
                   edgecolors=CAT[i], linewidths=1.1, marker="^", zorder=4)
    ax.set_title(title, fontsize=12, color=INK, pad=10)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var.)")
    ax.axhline(0, color=GRID, linewidth=1, zorder=1)
    ax.axvline(0, color=GRID, linewidth=1, zorder=1)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)

plot_panel(axes[0], Ptr_raw, Pte_raw, var_raw, "(a) Raw feature space\n(standardized MFCC/spectral features, no reference subtraction)")
plot_panel(axes[1], Ptr_res, Pte_res, var_res, "(b) Residual (discriminative) space\n(reference-subtracted, normal-range, own-speaker residual)")

# shared legend below, plus a note on marker shape
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=9.5)
fig.text(0.5, -0.075, "Filled circle = train sample   |   Open triangle = test sample",
         ha="center", fontsize=9, color=INK2, style="italic")

fig.suptitle("2D PCA projection: raw features vs. reference-subtracted residuals", fontsize=13.5, color=INK, y=1.03)
fig.tight_layout()
fig.savefig("./fig_pca_raw_vs_residual.png", dpi=200, bbox_inches="tight")
print("saved fig_pca_raw_vs_residual.png")
print("raw var explained (PC1,PC2):", var_raw[:2])
print("residual var explained (PC1,PC2):", var_res[:2])
