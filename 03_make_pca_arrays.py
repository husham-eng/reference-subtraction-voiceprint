# NOTE (revision): not used in the revised paper. Test residuals here use each sample's TRUE
# speaker reference, i.e. test labels, so the projection is not a valid discrimination diagnostic.
"""
Builds the raw-feature and reference-subtracted-residual arrays used by
05_make_pca_figure.py. Must be run after 01_extract_features.py.

Each sample's residual uses its OWN true speaker label's leave-one-out
reference (mean-range normal-range reference, as used by the best
Reference-Subtraction variant) — i.e. z_c(x) where c = speaker(x).
"""
import numpy as np
import pandas as pd

df = pd.read_csv("features.csv")
feat_cols = [c for c in df.columns if c not in ("speaker", "split", "path", "filename")]
train = df[df.split == "train"].reset_index(drop=True)
test = df[df.split == "test"].reset_index(drop=True)
speakers = sorted(train.speaker.unique().tolist())

mu = train[feat_cols].mean()
sigma = train[feat_cols].std().replace(0, 1.0)
Xtr = ((train[feat_cols] - mu) / sigma).values
Xte = ((test[feat_cols] - mu) / sigma).values
ytr = train.speaker.values
yte = test.speaker.values

R_mean = {c: Xtr[ytr != c].mean(axis=0) for c in speakers}
R_std = {c: Xtr[ytr != c].std(axis=0) + 1e-6 for c in speakers}


def residual_own_label(X, y):
    out = np.zeros_like(X)
    for i in range(len(X)):
        c = y[i]
        out[i] = (X[i] - R_mean[c]) / R_std[c]
    return out


Ztr = residual_own_label(Xtr, ytr)
Zte = residual_own_label(Xte, yte)

np.save("pca_Xtr.npy", Xtr)
np.save("pca_Xte.npy", Xte)
np.save("pca_Ztr.npy", Ztr)
np.save("pca_Zte.npy", Zte)
np.save("pca_ytr.npy", ytr)
np.save("pca_yte.npy", yte)
print("saved pca_Xtr.npy, pca_Xte.npy, pca_Ztr.npy, pca_Zte.npy, pca_ytr.npy, pca_yte.npy")
