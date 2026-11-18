"""
SSW60 analysis on saved embeddings (CPU only, no re-embedding).

Input: embeddings.npz from the burds-ssw60-baseline kernel (attached via kernel_sources).
Covers:
  A1 weighted fusion       - p_img^w * p_aud^(1-w), w tuned on TRAIN out-of-fold predictions
  A2 confidence fusion     - weight each modality by its own confidence (max prob)
  A3 linear probes         - logistic regression on frozen embeddings (image, audio, joint,
                             and early fusion on concatenated [image, audio] pairs)
  A4 better matching       - k-NN voting with k chosen by cross-validation on train
  A5 error analysis        - confusion matrices, most-confused species pairs, per-species table
Every hyperparameter is chosen on the train split only; test is touched once for reporting.
Headline numbers get 95% bootstrap confidence intervals.
"""
import glob
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

T0 = time.time()
OUT = "/kaggle/working"
SEED = 0
rng = np.random.default_rng(SEED)


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


src = glob.glob("/kaggle/input/**/embeddings.npz", recursive=True)[0]
taxa_p = glob.glob("/kaggle/input/**/taxa.csv", recursive=True)
log(f"loading {src}")
Z = np.load(src, allow_pickle=True)  # our own file; id/split columns were saved as object arrays
L = Z["labels"]
NAME = dict(zip(pd.read_csv(taxa_p[0]).label, pd.read_csv(taxa_p[0]).common_name)) if taxa_p else {}
nm = lambda l: NAME.get(int(l), str(l))  # noqa: E731
K = len(L)
lab2idx = {int(l): i for i, l in enumerate(L)}


def norm(E, d=None):
    E = E.astype(np.float32)
    if d:
        E = E[:, :d]
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-12)


def yidx(y):
    return np.array([lab2idx[int(v)] for v in y])


img, aud = norm(Z["img"]), norm(Z["aud"])
yi, ya = yidx(Z["img_y"]), yidx(Z["aud_y"])
tri, tei = Z["img_split"] == "train", Z["img_split"] == "test"
tra, tea = Z["aud_split"] == "train", Z["aud_split"] == "test"
PTR, PTE = Z["pairs_train"], Z["pairs_test"]  # rows: (img_row, aud_row, label)
pj_tr, pj_te = norm(Z["pair_train"]), norm(Z["pair_test"])
log(f"img train/test {tri.sum()}/{tei.sum()}  aud {tra.sum()}/{tea.sum()}  pairs {len(PTR)}/{len(PTE)}")

CV = StratifiedKFold(5, shuffle=True, random_state=SEED)


# ----------------------------------------------------------------- scorers -> (n, K) probabilities
def softmax(S, t):
    S = S / t
    S = S - S.max(1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(1, keepdims=True)


def centroid_scores(Xtr, ytr, Xte):
    C = np.stack([Xtr[ytr == k].mean(0) if (ytr == k).any() else np.zeros(Xtr.shape[1]) for k in range(K)])
    C /= np.linalg.norm(C, axis=1, keepdims=True).clip(1e-12)
    return Xte @ C.T


def knn_probs(Xtr, ytr, Xte, k, t=0.05):
    S = Xte @ Xtr.T
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    P = np.zeros((len(Xte), K))
    w = np.exp(np.take_along_axis(S, top, 1) / t)
    for j in range(k):
        np.add.at(P, (np.arange(len(Xte)), ytr[top[:, j]]), w[:, j])
    return P / P.sum(1, keepdims=True)


def oof(fn, X, y):
    """Out-of-fold probabilities on train: fn(Xtr, ytr, Xva) -> probs."""
    P = np.zeros((len(X), K))
    for a, b in CV.split(X, y):
        P[b] = fn(X[a], y[a], X[b])
    return P


def acc(P, y, k=1):
    top = np.argsort(-P, 1)[:, :k]
    return float(np.mean((top == y[:, None]).any(1)))


def boot_ci(P, y, n=1000):
    hit = (P.argmax(1) == y).astype(float)
    b = rng.integers(0, len(y), (n, len(y)))
    s = hit[b].mean(1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def fit_lr(X, y, C):
    return LogisticRegression(C=C, max_iter=3000, n_jobs=-1).fit(X, y)


def tune_lr(X, y, grid=(0.3, 1, 3, 10, 30)):
    best = max(grid, key=lambda C: acc(cross_val_predict(
        LogisticRegression(C=C, max_iter=3000, n_jobs=-1), X, y, cv=CV, method="predict_proba"), y))
    return best


R = {"single": {}, "paired": {}, "hparams": {}}


def single_modality(name, X, y, tr, te):
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    out, probs = {}, {}
    # centroid (baseline): temperature tuned on train OOF for calibrated fusion later
    St = oof(lambda a, b, c: centroid_scores(a, b, c), Xtr, ytr)
    t_c = min([0.01, 0.02, 0.05, 0.1], key=lambda t: -np.mean(np.log(softmax(St, t)[np.arange(len(ytr)), ytr] + 1e-9)))
    probs["centroid"] = (softmax(St, t_c), softmax(centroid_scores(Xtr, ytr, Xte), t_c))
    # k-NN with k by CV
    kbest = max([1, 3, 5, 10, 20, 40], key=lambda k: acc(oof(lambda a, b, c: knn_probs(a, b, c, k), Xtr, ytr), ytr))
    probs["knn"] = (oof(lambda a, b, c: knn_probs(a, b, c, kbest), Xtr, ytr), knn_probs(Xtr, ytr, Xte, kbest))
    # linear probe
    C = tune_lr(Xtr, ytr)
    lr_oof = cross_val_predict(LogisticRegression(C=C, max_iter=3000, n_jobs=-1), Xtr, ytr, cv=CV,
                               method="predict_proba")
    m = fit_lr(Xtr, ytr, C)
    probs["linear_probe"] = (lr_oof, m.predict_proba(Xte))
    R["hparams"][name] = {"centroid_temp": t_c, "knn_k": int(kbest), "lr_C": C}
    for k, (_, Pte) in probs.items():
        out[k] = {"top1": acc(Pte, yte), "top5": acc(Pte, yte, 5), "ci95": boot_ci(Pte, yte), "n": int(len(yte))}
    log(f"{name}: " + "  ".join(f"{k}={v['top1']:.3f}" for k, v in out.items()) + f"  hp={R['hparams'][name]}")
    R["single"][name] = out
    return probs, m


log("A3/A4: single-modality probes")
img_probs, img_lr = single_modality("image", img, yi, tri, tei)
aud_probs, aud_lr = single_modality("audio", aud, ya, tra, tea)

# Map train-split OOF arrays back to full-row indexing so pairs can look them up.
def full_rows(P_oof, P_te, tr, te, n):
    F = np.zeros((n, K))
    F[np.where(tr)[0]] = P_oof
    F[np.where(te)[0]] = P_te
    return F


# ----------------------------------------------------------------- A1/A2 fusion on paired sets
log("A1/A2: fusion")
ytr_p, yte_p = yidx(PTR[:, 2]), yidx(PTE[:, 2])
W = np.linspace(0, 1, 21)


def fuse_weighted(Pi, Pa, w):
    return w * np.log(Pi + 1e-9) + (1 - w) * np.log(Pa + 1e-9)


def fuse_conf(Pi, Pa, g):
    ci, ca = Pi.max(1, keepdims=True) ** g, Pa.max(1, keepdims=True) ** g
    return (ci * np.log(Pi + 1e-9) + ca * np.log(Pa + 1e-9)) / (ci + ca)


for method in ("centroid", "knn", "linear_probe"):
    Fi = full_rows(*img_probs[method], tri, tei, len(img))
    Fa = full_rows(*aud_probs[method], tra, tea, len(aud))
    Pi_tr, Pa_tr = Fi[PTR[:, 0]], Fa[PTR[:, 1]]
    Pi_te, Pa_te = Fi[PTE[:, 0]], Fa[PTE[:, 1]]
    w = float(max(W, key=lambda w: acc(fuse_weighted(Pi_tr, Pa_tr, w), ytr_p)))
    g = float(max([0.5, 1, 2, 4, 8], key=lambda g: acc(fuse_conf(Pi_tr, Pa_tr, g), ytr_p)))
    rows = {"image_only": Pi_te, "audio_only": Pa_te,
            "equal_weight": fuse_weighted(Pi_te, Pa_te, 0.5),
            "weighted": fuse_weighted(Pi_te, Pa_te, w),
            "confidence": fuse_conf(Pi_te, Pa_te, g)}
    R["paired"][method] = {k: {"top1": acc(v, yte_p), "ci95": boot_ci(v, yte_p)} for k, v in rows.items()}
    R["hparams"][f"fusion_{method}"] = {"image_weight": w, "conf_gamma": g}
    log(f"{method}: " + "  ".join(f"{k}={v['top1']:.3f}" for k, v in R['paired'][method].items()) + f"  w={w} g={g}")
    if method == "linear_probe":
        best_fused = rows["weighted"]

# Joint single-embedding: linear probe on joint embeddings of train pairs.
Cj = tune_lr(pj_tr, ytr_p)
Pj = fit_lr(pj_tr, ytr_p, Cj).predict_proba(pj_te)
R["paired"]["joint_embedding_probe"] = {"top1": acc(Pj, yte_p), "ci95": boot_ci(Pj, yte_p), "C": Cj}
log(f"joint embedding probe: {acc(Pj, yte_p):.3f}")

# Early fusion: LR on [img, aud] concatenations. Train pairs can be resampled freely from train
# embeddings (any train image + any train clip of the same species), no GPU needed.
def sample_pairs(split_i, split_a, per_class):
    I, A, Y = [], [], []
    for k in range(K):
        ii = np.where(split_i & (yi == k))[0]
        aa = np.where(split_a & (ya == k))[0]
        if len(ii) and len(aa):
            I += list(rng.choice(ii, per_class))
            A += list(rng.choice(aa, per_class))
            Y += [k] * per_class
    return np.array(I), np.array(A), np.array(Y)


I, A, Y = sample_pairs(tri, tra, 200)
Xc_tr = np.hstack([img[I], aud[A]])
Xc_te = np.hstack([img[PTE[:, 0]], aud[PTE[:, 1]]])
Cc = 1.0
Pc = fit_lr(Xc_tr, Y, Cc).predict_proba(Xc_te)
R["paired"]["early_fusion_probe"] = {"top1": acc(Pc, yte_p), "ci95": boot_ci(Pc, yte_p),
                                     "train_pairs": int(len(Y)), "C": Cc}
log(f"early fusion probe: {acc(Pc, yte_p):.3f}")

# Matryoshka check for the best pipeline (linear probe + weighted fusion)
R["dims"] = {}
for d in (768, 256, 128):
    Xi, Xa = norm(Z["img"], d), norm(Z["aud"], d)
    Ci_, Ca_ = R["hparams"]["image"]["lr_C"], R["hparams"]["audio"]["lr_C"]
    pi = fit_lr(Xi[tri], yi[tri], Ci_).predict_proba(Xi[PTE[:, 0]])
    pa = fit_lr(Xa[tra], ya[tra], Ca_).predict_proba(Xa[PTE[:, 1]])
    w = R["hparams"]["fusion_linear_probe"]["image_weight"]
    R["dims"][d] = {"image_probe": acc(pi, yte_p), "audio_probe": acc(pa, yte_p),
                    "weighted_fusion": acc(fuse_weighted(pi, pa, w), yte_p)}
    log(f"d={d}: {R['dims'][d]}")

# ----------------------------------------------------------------- A5 error analysis
log("A5: error analysis")
Pimg_te = img_probs["linear_probe"][1]


def confusion(P, y):
    M = np.zeros((K, K), int)
    np.add.at(M, (y, P.argmax(1)), 1)
    return M


def top_confusions(M, n=12):
    off = [(M[a, b] / max(M[a].sum(), 1), a, b) for a in range(K) for b in range(K) if a != b and M[a, b]]
    return [{"true": nm(L[a]), "predicted": nm(L[b]), "rate": float(r), "count": int(M[a, b])}
            for r, a, b in sorted(off, reverse=True)[:n]]


M_img = confusion(Pimg_te, yi[tei])
M_aud = confusion(aud_probs["linear_probe"][1], ya[tea])
M_fus = confusion(best_fused, yte_p)
R["confusions"] = {"image": top_confusions(M_img), "audio": top_confusions(M_aud), "fused": top_confusions(M_fus)}

Pa_te_pairs = full_rows(*aud_probs["linear_probe"], tra, tea, len(aud))[PTE[:, 1]]
Pi_te_pairs = full_rows(*img_probs["linear_probe"], tri, tei, len(img))[PTE[:, 0]]
per = []
for k in range(K):
    m = yte_p == k
    if m.any():
        per.append({"species": nm(L[k]), "n": int(m.sum()),
                    "image": float(np.mean(Pi_te_pairs[m].argmax(1) == k)),
                    "audio": float(np.mean(Pa_te_pairs[m].argmax(1) == k)),
                    "fused": float(np.mean(best_fused[m].argmax(1) == k))})
R["per_species"] = sorted(per, key=lambda p: p["fused"] - p["image"])

order = np.argsort([nm(l) for l in L])
for M, title, fn in ((M_img, "Photo (linear probe)", "confusion_image.png"),
                     (M_aud, "Song (linear probe)", "confusion_audio.png"),
                     (M_fus, "Photo + song (weighted fusion)", "confusion_fused.png")):
    Mn = M / M.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(13, 12))
    ax.imshow(Mn[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1)
    names = [nm(L[i]) for i in order]
    ax.set_xticks(range(K)); ax.set_xticklabels(names, rotation=90, fontsize=6)  # noqa: E702
    ax.set_yticks(range(K)); ax.set_yticklabels(names, fontsize=6)  # noqa: E702
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)  # noqa: E702
    fig.tight_layout(); fig.savefig(f"{OUT}/{fn}", dpi=120); plt.close(fig)  # noqa: E702

# ----------------------------------------------------------------- write
R = json.loads(json.dumps(R, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
json.dump(R, open(f"{OUT}/analysis.json", "w"), indent=2)
pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
ci = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"  # noqa: E731
md = ["# SSW60 analysis on frozen EmbeddingGemma 2 embeddings", "",
      "All hyperparameters chosen by 5-fold CV on the train split; 95% bootstrap CIs on test.", "",
      "## Single modality (full test sets)", "",
      "| modality | method | top-1 | 95% CI | top-5 | n |", "|---|---|---|---|---|---|"]
for mod in ("image", "audio"):
    for k, v in R["single"][mod].items():
        md.append(f"| {mod} | {k} | {pct(v['top1'])} | {ci(v['ci95'])} | {pct(v['top5'])} | {v['n']} |")
md += ["", f"## Photo + song (paired test set, n={len(yte_p)})", "",
       "| per-modality scorer | image only | audio only | equal weight | weighted | confidence |",
       "|---|---|---|---|---|---|"]
for k in ("centroid", "knn", "linear_probe"):
    p = R["paired"][k]
    md.append(f"| {k} | " + " | ".join(pct(p[c]["top1"]) for c in
                                         ("image_only", "audio_only", "equal_weight", "weighted", "confidence")) + " |")
bf = R["paired"]["linear_probe"]["weighted"]
md += ["", f"- Best fused (linear probe + weighted, image weight "
           f"{R['hparams']['fusion_linear_probe']['image_weight']:.2f}): **{pct(bf['top1'])}** {ci(bf['ci95'])}",
       f"- Image-only on same pairs: {pct(R['paired']['linear_probe']['image_only']['top1'])} "
       f"{ci(R['paired']['linear_probe']['image_only']['ci95'])}",
       f"- Joint single-embedding probe: {pct(R['paired']['joint_embedding_probe']['top1'])}",
       f"- Early fusion probe on [image, audio]: {pct(R['paired']['early_fusion_probe']['top1'])}", "",
       "## Embedding size (linear probe + weighted fusion, paired test)", "",
       "| dims | image | audio | fused |", "|---|---|---|---|"]
for d, v in R["dims"].items():
    md.append(f"| {d} | {pct(v['image_probe'])} | {pct(v['audio_probe'])} | {pct(v['weighted_fusion'])} |")
for mod in ("image", "audio", "fused"):
    md += ["", f"## Most confused ({mod})", "", "| true | predicted as | rate |", "|---|---|---|"]
    md += [f"| {c['true']} | {c['predicted']} | {pct(c['rate'])} ({c['count']}) |" for c in R["confusions"][mod][:8]]
md += ["", "## Where song helps / hurts most (paired, linear probe)", "",
       "| species | image | audio | fused |", "|---|---|---|---|"]
for p in R["per_species"][:6] + R["per_species"][-6:]:
    md.append(f"| {p['species']} | {pct(p['image'])} | {pct(p['audio'])} | {pct(p['fused'])} |")
md += ["", f"Hyperparameters: `{json.dumps(R['hparams'])}`"]
open(f"{OUT}/analysis.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
log("done")
