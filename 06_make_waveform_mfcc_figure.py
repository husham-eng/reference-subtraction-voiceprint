import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SR = 16000
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
seq_cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

files = {
    1: "./train/speaker01/person1-1.ogg",
    2: "./train/speaker02/person-2-1.ogg",
    3: "./train/speaker03/person-3-1.ogg",
    4: "./train/speaker04/person-4-1.ogg",
    5: "./train/speaker05/person-5-1.ogg",
}

fig, axes = plt.subplots(2, 5, figsize=(15.5, 5.6), sharex=False)

for i, (spk, path) in enumerate(files.items()):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    t = np.arange(len(y)) / sr

    ax_w = axes[0, i]
    ax_w.plot(t, y, color=CAT[i], linewidth=0.8)
    ax_w.set_title(f"Speaker {spk}", fontsize=11.5, color=INK, pad=8)
    ax_w.set_xlim(0, max(t.max(), 0.1))
    ax_w.set_ylim(-1.05, 1.05)
    ax_w.axhline(0, color=GRID, linewidth=0.8, zorder=1)
    if i == 0:
        ax_w.set_ylabel("Amplitude", fontsize=9.5)
    ax_w.set_xlabel("Time (s)", fontsize=9)
    for spine in ["top", "right"]:
        ax_w.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_w.spines[spine].set_color(GRID)
    ax_w.tick_params(labelsize=8)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=256)
    # per-coefficient (row-wise) z-score, for visualization contrast only —
    # coefficient 1 (log-energy) has a much larger raw dynamic range than 2-13
    # and would otherwise wash out their texture on a shared color scale.
    mfcc_viz = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
    ax_m = axes[1, i]
    im = ax_m.imshow(mfcc_viz, aspect="auto", origin="lower", cmap=seq_cmap,
                      vmin=-2.5, vmax=2.5, extent=[0, t.max(), 1, 13])
    if i == 0:
        ax_m.set_ylabel("MFCC coeff.", fontsize=9.5)
    ax_m.set_xlabel("Time (s)", fontsize=9)
    ax_m.tick_params(labelsize=8)
    for spine in ax_m.spines.values():
        spine.set_visible(False)

fig.suptitle("Audio waveforms (top) and MFCC spectrograms (bottom) — one representative utterance per speaker",
             fontsize=13.5, color=INK, y=1.03)

cbar_ax = fig.add_axes([1.005, 0.10, 0.012, 0.35])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8, color=INK2, labelcolor=INK2)
cbar.set_label("MFCC (per-coeff. z-score)", fontsize=9, color=INK2)
cbar.outline.set_visible(False)

fig.tight_layout()
fig.savefig("./fig_waveform_mfcc.png", dpi=200, bbox_inches="tight")
print("saved fig_waveform_mfcc.png")
