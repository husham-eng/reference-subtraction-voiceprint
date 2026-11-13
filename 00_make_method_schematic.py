"""
Generates the method-schematic diagram (fig_method_schematic.png) used as
Figure 1 in both papers. Has no data dependency — pure diagram.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#0b0b0b"; INK2 = "#52514e"; SURFACE = "#fcfcfb"; GRID = "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")


def box(x, y, w, h, text, color, fontsize=10.5, textcolor="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                        linewidth=0, facecolor=color)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=textcolor,
            fontsize=fontsize, wrap=True, linespacing=1.4)


def arrow(x1, y1, x2, y2, color=INK2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.6, color=color)
    ax.add_patch(a)


box(0.2, 3.2, 2.3, 1.0, "Training utterances\nof all OTHER speakers", CAT[0])
arrow(2.5, 3.7, 3.3, 3.7)

box(3.3, 3.2, 2.3, 1.0, "Reference R_c\n(normal range: mean ± std\nof the other speakers)", CAT[2])
arrow(4.45, 3.2, 4.45, 2.4)

box(0.2, 1.7, 2.3, 1.0, "Candidate utterance x\n(claimed speaker c)", CAT[1])
arrow(2.5, 2.2, 3.3, 2.55)

box(3.3, 1.7, 2.3, 1.0, "Representation:\nresidual = (x − R_c) / std_c", "#5f5e58")
arrow(5.6, 2.2, 6.4, 2.2)

box(6.4, 1.7, 2.2, 1.0, "Compare to speaker c's\nfingerprint template F_c", CAT[3], textcolor=INK)
arrow(8.6, 2.2, 9.4, 2.2)

box(9.4, 1.7, 1.4, 1.0, "Accept /\nReject c", CAT[4], textcolor=INK)

ax.text(5.5, 0.9, "Fingerprint = what remains after removing the common (shared) part — the discriminative residual unique to speaker c",
        ha="center", va="center", fontsize=9.5, color=INK2, style="italic")

fig.tight_layout()
fig.savefig("./fig_method_schematic.png", dpi=200, bbox_inches="tight")
print("saved fig_method_schematic.png")
