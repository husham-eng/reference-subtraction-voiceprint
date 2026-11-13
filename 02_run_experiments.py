"""
Full ablation: compares
  - Baseline nearest-centroid (raw, no reference subtraction) x {cosine, euclidean}
  - Reference-Subtraction, mean-only reference x {cosine, euclidean}
  - Reference-Subtraction, normal-range reference (mean+std of the "others" population) x {cosine, euclidean, mahalanobis-diag}
All fit on TRAIN (425), evaluated on TEST (180), features standardized using TRAIN stats only.
"""
import numpy as np, pandas as pd, json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chi2

df = pd.read_csv("./features.csv")
feat_cols = [c for c in df.columns if c not in ("speaker","split","path","filename")]
train = df[df.split=="train"].reset_index(drop=True)
test = df[df.split=="test"].reset_index(drop=True)
speakers = sorted(train.speaker.unique().tolist())

mu = train[feat_cols].mean(); sigma = train[feat_cols].std().replace(0,1.0)
Xtr = ((train[feat_cols]-mu)/sigma).values
Xte = ((test[feat_cols]-mu)/sigma).values
ytr = train.speaker.values
yte = test.speaker.values

# ---------- References ----------
R_mean = {c: Xtr[ytr!=c].mean(axis=0) for c in speakers}
R_std  = {c: Xtr[ytr!=c].std(axis=0)+1e-6 for c in speakers}
C      = {c: Xtr[ytr==c].mean(axis=0) for c in speakers}  # plain class centroid (no subtraction)

def resid_mean_only(x, c):
    return x - R_mean[c]

def resid_normal_range(x, c):
    return (x - R_mean[c]) / R_std[c]

F_mean_only    = {c: resid_mean_only(Xtr[ytr==c], c).mean(axis=0) for c in speakers}
F_normal_range = {c: resid_normal_range(Xtr[ytr==c], c).mean(axis=0) for c in speakers}
Fstd_normal_range = {c: resid_normal_range(Xtr[ytr==c], c).std(axis=0)+1e-6 for c in speakers}

def predict(X, score_fn, maximize=True):
    preds=[]
    for x in X:
        s = {c: score_fn(x, c) for c in speakers}
        preds.append(max(s, key=s.get) if maximize else min(s, key=s.get))
    return np.array(preds)

methods = {}

# Baseline: nearest centroid, cosine
methods["Baseline-Centroid-Cosine"] = predict(
    Xte, lambda x,c: cosine_similarity(x.reshape(1,-1), C[c].reshape(1,-1))[0,0], maximize=True)

# Baseline: nearest centroid, euclidean
methods["Baseline-Centroid-Euclid"] = predict(
    Xte, lambda x,c: -np.linalg.norm(x - C[c]), maximize=True)

# RefSub mean-only, cosine
methods["RefSub-MeanOnly-Cosine"] = predict(
    Xte, lambda x,c: cosine_similarity(resid_mean_only(x,c).reshape(1,-1), F_mean_only[c].reshape(1,-1))[0,0], maximize=True)

# RefSub mean-only, euclidean  (algebraically identical to Baseline-Centroid-Euclid -- kept to demonstrate this)
methods["RefSub-MeanOnly-Euclid"] = predict(
    Xte, lambda x,c: -np.linalg.norm(resid_mean_only(x,c) - F_mean_only[c]), maximize=True)

# RefSub normal-range (mean+std of others), cosine
methods["RefSub-NormalRange-Cosine"] = predict(
    Xte, lambda x,c: cosine_similarity(resid_normal_range(x,c).reshape(1,-1), F_normal_range[c].reshape(1,-1))[0,0], maximize=True)

# RefSub normal-range, euclidean
methods["RefSub-NormalRange-Euclid"] = predict(
    Xte, lambda x,c: -np.linalg.norm(resid_normal_range(x,c) - F_normal_range[c]), maximize=True)

# RefSub normal-range, Mahalanobis-diagonal (double-normalized)
methods["RefSub-NormalRange-MahalanobisDiag"] = predict(
    Xte, lambda x,c: -np.sum(((resid_normal_range(x,c) - F_normal_range[c]) / Fstd_normal_range[c])**2), maximize=True)

# ---------- Evaluate all ----------
summary = []
cms = {}
for name, pred in methods.items():
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro")
    cm = confusion_matrix(yte, pred, labels=speakers)
    cms[name] = cm.tolist()
    summary.append({"method": name, "accuracy": round(acc,4), "macro_f1": round(f1m,4)})

summary_df = pd.DataFrame(summary).sort_values("accuracy", ascending=False)
print(summary_df.to_string(index=False))

# ---------- McNemar test: best RefSub variant vs best Baseline variant ----------
def mcnemar(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = np.sum(correct_a & ~correct_b)   # a right, b wrong
    c = np.sum(~correct_a & correct_b)   # a wrong, b right
    if b + c == 0:
        return 0.0, 1.0, b, c
    stat = (abs(b-c)-1)**2 / (b+c)
    p = 1 - chi2.cdf(stat, df=1)
    return stat, p, b, c

# Compare an explicitly named pair. (The submitted version selected the "best" variants by
# sorting on accuracy; with several variants tied at 0.850 this picked NormalRange-Euclid
# instead of the MahalanobisDiag variant discussed in the paper.) Exact test: 09_reviewer_checks.py
best_refsub_name = "RefSub-NormalRange-MahalanobisDiag"
best_base_name = "Baseline-Centroid-Euclid"
stat, p, b, c = mcnemar(yte, methods[best_refsub_name], methods[best_base_name])
print(f"\nMcNemar test: {best_refsub_name} vs {best_base_name}")
print(f"disagreements: refsub-only-correct={b}, baseline-only-correct={c}, chi2={stat:.4f}, p={p:.4f}")

# check algebraic-equivalence claim empirically
identical = np.array_equal(methods["RefSub-MeanOnly-Euclid"], methods["Baseline-Centroid-Euclid"])
print(f"\nRefSub-MeanOnly-Euclid identical to Baseline-Centroid-Euclid predictions: {identical}")

out = {
    "summary": summary,
    "confusion_matrices": cms,
    "speakers": speakers,
    "n_train": len(train),
    "n_test": len(test),
    "n_features": len(feat_cols),
    "feat_cols": feat_cols,
    "mcnemar_best_refsub_vs_best_baseline": {
        "refsub_method": best_refsub_name, "baseline_method": best_base_name,
        "chi2": stat, "p_value": p, "refsub_only_correct": int(b), "baseline_only_correct": int(c)
    },
    "meanonly_euclid_equals_baseline_centroid_euclid": bool(identical),
    "per_method_classification_report": {
        name: classification_report(yte, pred, labels=speakers, output_dict=True, zero_division=0)
        for name, pred in methods.items()
    }
}
with open("./experiments_results.json","w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

# save predictions
pred_df = test[["filename","speaker"]].copy()
for name, pred in methods.items():
    pred_df[name] = pred
pred_df.to_csv("./all_predictions.csv", index=False)
print("\nSaved experiments_results.json and all_predictions.csv")
