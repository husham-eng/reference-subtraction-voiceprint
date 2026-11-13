"""
Reviewer checks (ICPIC 2026, paper 108):
  M1  Does the normal-range Mahalanobis variant reduce to a plain class-dependent
      diagonal Mahalanobis distance? -> add that baseline and compare predictions.
      Also: what does normal-range Euclidean reduce to?
  M2  Exact McNemar (paired) between the variants actually shown in Fig. 5.
  M4  Paired stratified bootstrap CI for macro-F1 and accuracy differences;
      train-only cross-validated model selection.
Run after 02_run_experiments.py (uses features.csv and the same preprocessing).
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("features.csv")
feat = [c for c in df.columns if c not in ("speaker", "split", "path", "filename")]
tr = df[df.split == "train"].reset_index(drop=True)
te = df[df.split == "test"].reset_index(drop=True)
spk = sorted(tr.speaker.unique())

def standardize(train, test):
    mu = train[feat].mean(); sd = train[feat].std().replace(0, 1.0)
    return ((train[feat] - mu) / sd).values, ((test[feat] - mu) / sd).values

Xtr, Xte = standardize(tr, te)
ytr, yte = tr.speaker.values, te.speaker.values
EPS = 1e-6

def fit(X, y):
    m = {}
    for c in spk:
        oth, own = X[y != c], X[y == c]
        Rm, Rs = oth.mean(0), oth.std(0) + EPS
        V = (own - Rm) / Rs
        m[c] = dict(Rm=Rm, Rs=Rs, C=own.mean(0), Sd=own.std(0) + EPS,
                    F=V.mean(0), Fs=V.std(0) + EPS, Uo=(own - Rm), G=(own - Rm).mean(0))
    return m

def predict(X, m, rule):
    out = []
    for x in X:
        s = {}
        for c in spk:
            p = m[c]
            v = (x - p["Rm"]) / p["Rs"]
            if rule == "base_euclid":   s[c] = -np.sum((x - p["C"]) ** 2)
            elif rule == "base_maha":   s[c] = -np.sum(((x - p["C"]) / p["Sd"]) ** 2)       # NEW baseline
            elif rule == "base_maha_ref_std": s[c] = -np.sum(((x - p["C"]) / p["Rs"]) ** 2)  # what NR-Euclid reduces to
            elif rule == "rs_nr_euclid": s[c] = -np.sum((v - p["F"]) ** 2)
            elif rule == "rs_nr_maha":  s[c] = -np.sum(((v - p["F"]) / p["Fs"]) ** 2)
            elif rule == "rs_nr_cos":   s[c] = v @ p["F"] / (np.linalg.norm(v) * np.linalg.norm(p["F"]))
            elif rule == "base_cos":    s[c] = x @ p["C"] / (np.linalg.norm(x) * np.linalg.norm(p["C"]))
            elif rule == "rs_mo_cos":
                u = x - p["Rm"]; s[c] = u @ p["G"] / (np.linalg.norm(u) * np.linalg.norm(p["G"]))
        out.append(max(s, key=s.get))
    return np.array(out)

m = fit(Xtr, ytr)
rules = ["base_cos", "base_euclid", "base_maha", "base_maha_ref_std",
         "rs_mo_cos", "rs_nr_cos", "rs_nr_euclid", "rs_nr_maha"]
P = {r: predict(Xte, m, r) for r in rules}
res = {}
print("== Test-set results ==")
for r in rules:
    res[r] = dict(acc=accuracy_score(yte, P[r]), f1=f1_score(yte, P[r], average="macro"))
    print(f"{r:<20} acc={res[r]['acc']:.4f}  macroF1={res[r]['f1']:.4f}")

same = lambda a, b: int(np.sum(P[a] == P[b]))
print("\n== M1: equivalences (identical predictions out of 180) ==")
print("RefSub NR-Mahalanobis vs plain diagonal Mahalanobis :", same("rs_nr_maha", "base_maha"))
print("RefSub NR-Euclid vs centroid distance scaled by others' std:", same("rs_nr_euclid", "base_maha_ref_std"))
# max absolute score difference, to show the cancellation is exact up to the epsilon
x = Xte[0]
for c in spk:
    p = m[c]; v = (x - p["Rm"]) / p["Rs"]
    a = np.sum(((v - p["F"]) / p["Fs"]) ** 2); b = np.sum(((x - p["C"]) / p["Sd"]) ** 2)
    print(f"  speaker {c}: RefSub-Maha={a:.6f}  plain-Maha={b:.6f}  rel.diff={abs(a-b)/b:.2e}")

def mcnemar(a, b):
    ca, cb = P[a] == yte, P[b] == yte
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    both, neither = int(np.sum(ca & cb)), int(np.sum(~ca & ~cb))
    exact = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    cc = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0.0
    return dict(a_only=n01, b_only=n10, both=both, neither=neither, p_exact=exact,
                chi2_cc=cc, p_chi2_cc=1 - chi2.cdf(cc, 1))

print("\n== M2: McNemar (paired) ==")
mc = {}
for a, b in [("rs_nr_maha", "base_euclid"), ("rs_nr_euclid", "base_euclid"), ("rs_nr_maha", "base_maha")]:
    mc[f"{a}_vs_{b}"] = r = mcnemar(a, b)
    print(f"{a} vs {b}: {r}")

print("\n== M4: paired stratified bootstrap (B=10000) of differences vs base_euclid ==")
rng = np.random.default_rng(2026)
idx_by = {c: np.where(yte == c)[0] for c in spk}
B = 10000
boot = {}
for r in ["rs_nr_maha", "rs_nr_euclid", "base_maha"]:
    dF, dA = [], []
    for _ in range(B):
        idx = np.concatenate([rng.choice(idx_by[c], len(idx_by[c]), replace=True) for c in spk])
        y = yte[idx]
        dF.append(f1_score(y, P[r][idx], average="macro") - f1_score(y, P["base_euclid"][idx], average="macro"))
        dA.append(np.mean(P[r][idx] == y) - np.mean(P["base_euclid"][idx] == y))
    dF, dA = np.array(dF), np.array(dA)
    boot[r] = dict(dF1=res[r]["f1"] - res["base_euclid"]["f1"], dF1_ci=list(np.percentile(dF, [2.5, 97.5])),
                   dAcc=res[r]["acc"] - res["base_euclid"]["acc"], dAcc_ci=list(np.percentile(dA, [2.5, 97.5])))
    print(r, {k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for k, v in boot[r].items()})
# per-method macro-F1 CI
f1ci = {}
shared = [np.concatenate([rng.choice(idx_by[c], len(idx_by[c]), replace=True) for c in spk]) for _ in range(2000)]
for r in rules:   # same resamples for every method, so identical predictions give identical intervals
    vals = [f1_score(yte[idx], P[r][idx], average="macro") for idx in shared]
    f1ci[r] = list(np.percentile(vals, [2.5, 97.5]))
print("macro-F1 95% CI per method:", {k: np.round(v, 3).tolist() for k, v in f1ci.items()})

print("\n== M4: model selection on TRAIN only (5-fold stratified CV, 10 repeats) ==")
cv = {r: [] for r in rules}
for rep in range(10):
    skf = StratifiedKFold(5, shuffle=True, random_state=rep)
    for a, b in skf.split(tr[feat], ytr):
        Xa, Xb = standardize(tr.iloc[a], tr.iloc[b])
        mm = fit(Xa, ytr[a])
        for r in rules:
            cv[r].append(f1_score(ytr[b], predict(Xb, mm, r), average="macro"))
cvs = {r: (float(np.mean(v)), float(np.std(v))) for r, v in cv.items()}
for r, (mu, sd) in sorted(cvs.items(), key=lambda t: -t[1][0]):
    print(f"{r:<20} CV macro-F1 = {mu:.4f} ± {sd:.4f}")

json.dump(dict(test=res, mcnemar=mc, bootstrap=boot, f1_ci=f1ci, cv_train=cvs,
               equiv=dict(nr_maha_vs_plain_maha=same("rs_nr_maha", "base_maha"),
                          nr_euclid_vs_ref_std_scaled=same("rs_nr_euclid", "base_maha_ref_std"))),
          open("reviewer_checks.json", "w"), indent=2, default=float)
pd.DataFrame({"filename": te.filename, "speaker": yte, **{r: P[r] for r in rules}}).to_csv("predictions_all_rules.csv", index=False)
print("\nsaved reviewer_checks.json, predictions_all_rules.csv")
