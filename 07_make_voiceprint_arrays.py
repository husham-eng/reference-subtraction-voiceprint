"""
Builds the per-speaker raw centroid (C_c) and normal-range fingerprint (F_c)
arrays used by 08_make_voiceprint_figures.py. Must be run after
01_extract_features.py.
"""
import numpy as np
import pandas as pd

df = pd.read_csv("features.csv")
feat_cols = [c for c in df.columns if c not in ("speaker", "split", "path", "filename")]
train = df[df.split == "train"].reset_index(drop=True)
speakers = sorted(train.speaker.unique().tolist())

mu = train[feat_cols].mean()
sigma = train[feat_cols].std().replace(0, 1.0)
Xtr = ((train[feat_cols] - mu) / sigma).values
ytr = train.speaker.values

R_mean = {c: Xtr[ytr != c].mean(axis=0) for c in speakers}
R_std = {c: Xtr[ytr != c].std(axis=0) + 1e-6 for c in speakers}
C = {c: Xtr[ytr == c].mean(axis=0) for c in speakers}  # raw centroid
F = {c: ((Xtr[ytr == c] - R_mean[c]) / R_std[c]).mean(axis=0) for c in speakers}  # fingerprint

np.save("fp_F.npy", np.array([F[c] for c in speakers]))
np.save("fp_C.npy", np.array([C[c] for c in speakers]))
with open("fp_feat_cols.txt", "w") as f:
    f.write("\n".join(feat_cols))
print("saved fp_F.npy, fp_C.npy, fp_feat_cols.txt")
