"""
Clip-length sweep evaluation (CPU only): how many seconds of audio should we embed?

Loudest N seconds (energy above 1 kHz) for N in 1, 2, 3, 5, 7 and the full 10 s clip, at normal
and half speed. 3 s and 10 s come from burds-ssw60-embed (embed_v3.npz); 1/2/5/7 s from
burds-ssw60-audiolen (embed_audiolen.npz). Same protocol as before: logistic regression tuned by
5-fold CV on train, test used once, 95% bootstrap CIs. Then the best feature combination is
fused with photos on the run-2 paired test set.
"""
import glob
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

T0 = time.time()
OUT = "/kaggle/working"
rng = np.random.default_rng(0)
CV = StratifiedKFold(5, shuffle=True, random_state=0)
GRID = (3, 10, 30, 100, 300)


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def find(name):
    return glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]


A = np.load(find("embed_audiolen.npz"), allow_pickle=True)
V = np.load(find("embed_v3.npz"), allow_pickle=True)
B = np.load(find("embeddings.npz"), allow_pickle=True)
L = B["labels"]
K = len(L)
lab2idx = {int(l): i for i, l in enumerate(L)}
yidx = lambda y: np.array([lab2idx[int(v)] for v in y])  # noqa: E731
assert (A["aud_id"].astype(str) == V["aud_id"].astype(str)).all(), "audio row order differs between runs"


def norm(E):
    E = np.asarray(E, np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-12)


def acc(P, y, k=1):
    return float(np.mean((np.argsort(-P, 1)[:, :k] == y[:, None]).any(1)))


def ci(P, y, n=1000):
    hit = (P.argmax(1) == y).astype(float)
    s = hit[rng.integers(0, len(y), (n, len(y)))].mean(1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def lr(C):
    return LogisticRegression(C=C, max_iter=4000, n_jobs=-1)


def probe(Xtr, ytr, Xte):
    scores = {C: cross_val_predict(lr(C), Xtr, ytr, cv=CV, method="predict_proba") for C in GRID}
    C = max(GRID, key=lambda c: acc(scores[c], ytr))
    return scores[C], lr(C).fit(Xtr, ytr).predict_proba(Xte), C


ya = yidx(A["aud_y"])
atr, ate = A["aud_split"] == "train", A["aud_split"] == "test"

# feature name -> (seconds, speed) ; full clip = 10 s
feats = {}
for n in (1, 2, 5, 7):
    feats[(n, "normal")] = norm(A[f"aud_trim{n}"])
    feats[(n, "half")] = norm(A[f"aud_slow2_trim{n}"])
feats[(3, "normal")], feats[(3, "half")] = norm(V["aud_trim3"]), norm(V["aud_slow2_trim3"])
feats[(10, "normal")], feats[(10, "half")] = norm(V["aud_base"]), norm(V["aud_slow2"])

R = {"length_sweep": {}, "combos": {}, "fusion": {}}
probs = {}
for key in sorted(feats):
    oof, Pte, C = probe(feats[key][atr], ya[atr], feats[key][ate])
    probs[key] = (oof, Pte)
    R["length_sweep"][f"{key[0]}s_{key[1]}"] = {"seconds": key[0], "speed": key[1], "top1": acc(Pte, ya[ate]),
                                                "top5": acc(Pte, ya[ate], 5), "ci95": ci(Pte, ya[ate]), "C": C}
    log(f"{key}: {acc(Pte, ya[ate]):.3f}")

# Combinations: previous best ("all five" from step B) vs. adding the best lengths.
prev_all = np.hstack([norm(V[f"aud_{v}"]) for v in ("base", "trim3", "slow2", "slow2_trim3", "win3")])
best_half = max((k for k in feats if k[1] == "half"), key=lambda k: R["length_sweep"][f"{k[0]}s_half"]["top1"])
best_norm = max((k for k in feats if k[1] == "normal"), key=lambda k: R["length_sweep"][f"{k[0]}s_normal"]["top1"])
combos = {
    "step B: all five (previous best)": prev_all,
    f"half speed, all lengths": np.hstack([feats[(n, "half")] for n in (1, 2, 3, 5, 7, 10)]),
    f"normal speed, all lengths": np.hstack([feats[(n, "normal")] for n in (1, 2, 3, 5, 7, 10)]),
    "both speeds, all lengths": np.hstack([feats[k] for k in sorted(feats)]),
    f"best normal ({best_norm[0]}s) + best half ({best_half[0]}s) + full both speeds":
        np.hstack([feats[best_norm], feats[best_half], feats[(10, "normal")], feats[(10, "half")]]),
}
for name, X in combos.items():
    oof, Pte, C = probe(X[atr], ya[atr], X[ate])
    probs[name] = (oof, Pte)
    R["combos"][name] = {"dims": int(X.shape[1]), "top1": acc(Pte, ya[ate]), "top5": acc(Pte, ya[ate], 5),
                         "ci95": ci(Pte, ya[ate]), "C": C}
    log(f"combo {name}: {acc(Pte, ya[ate]):.3f}")

# Fusion with photos (run-2 paired test set), for the previous best and the new best combo.
best_combo = max(R["combos"], key=lambda k: R["combos"][k]["top1"])
img, yi = norm(B["img"]), yidx(B["img_y"])
itr, ite = B["img_split"] == "train", B["img_split"] == "test"
i_oof, i_te, _ = probe(img[itr], yi[itr], img[ite])
Fi = np.zeros((len(img), K)); Fi[np.where(itr)[0]] = i_oof; Fi[np.where(ite)[0]] = i_te  # noqa: E702
row_of = {a: i for i, a in enumerate(A["aud_id"].astype(str))}
b2a = np.array([row_of[a] for a in B["aud_id"].astype(str)])
PTR, PTE = B["pairs_train"], B["pairs_test"]
ytr_p, yte_p = yidx(PTR[:, 2]), yidx(PTE[:, 2])


def fuse_conf(Pi, Pa, g):
    a, b = Pi.max(1, keepdims=True) ** g, Pa.max(1, keepdims=True) ** g
    return (a * np.log(Pi + 1e-9) + b * np.log(Pa + 1e-9)) / (a + b)


for name in dict.fromkeys(["step B: all five (previous best)", best_combo]):
    Fa = np.zeros((len(ya), K)); Fa[np.where(atr)[0]] = probs[name][0]; Fa[np.where(ate)[0]] = probs[name][1]  # noqa: E702
    Pi_tr, Pa_tr, Pi_te, Pa_te = Fi[PTR[:, 0]], Fa[b2a[PTR[:, 1]]], Fi[PTE[:, 0]], Fa[b2a[PTE[:, 1]]]
    g = max([0.5, 1, 2, 4, 8], key=lambda g: acc(fuse_conf(Pi_tr, Pa_tr, g), ytr_p))
    Pf = fuse_conf(Pi_te, Pa_te, g)
    hi, hf = (Pi_te.argmax(1) == yte_p).astype(float), (Pf.argmax(1) == yte_p).astype(float)
    bidx = rng.integers(0, len(yte_p), (1000, len(yte_p)))
    d = (hf[bidx] - hi[bidx]).mean(1)
    R["fusion"][name] = {"image_only": acc(Pi_te, yte_p), "audio_only": acc(Pa_te, yte_p),
                         "fused": acc(Pf, yte_p), "fused_ci95": ci(Pf, yte_p), "gamma": g,
                         "gain_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    log(f"fusion {name}: {R['fusion'][name]}")

R = json.loads(json.dumps(R, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
json.dump(R, open(f"{OUT}/audiolen.json", "w"), indent=2)
pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
cis = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"  # noqa: E731
md = ["# How many seconds of audio? (loudest N seconds, linear probe, audio test n=%d)" % int(ate.sum()), "",
      "| seconds | normal speed | 95% CI | half speed | 95% CI |", "|---|---|---|---|---|"]
for n in (1, 2, 3, 5, 7, 10):
    a, b = R["length_sweep"][f"{n}s_normal"], R["length_sweep"][f"{n}s_half"]
    md.append(f"| {n if n < 10 else '10 (full clip)'} | {pct(a['top1'])} | {cis(a['ci95'])} | "
              f"{pct(b['top1'])} | {cis(b['ci95'])} |")
md += ["", "## Combining lengths", "", "| features | dims | top-1 | 95% CI | top-5 |", "|---|---|---|---|---|"]
for k, r in sorted(R["combos"].items(), key=lambda kv: -kv[1]["top1"]):
    md.append(f"| {k} | {r['dims']} | {pct(r['top1'])} | {cis(r['ci95'])} | {pct(r['top5'])} |")
md += ["", "## Photo + song (paired test, n=%d, confidence fusion)" % len(yte_p), "",
       "| audio features | image only | audio only | fused | gain vs image (95% CI) |", "|---|---|---|---|---|"]
for k, r in R["fusion"].items():
    g = r["gain_ci95"]
    md.append(f"| {k} | {pct(r['image_only'])} | {pct(r['audio_only'])} | **{pct(r['fused'])}** | "
              f"[{100 * g[0]:+.1f}, {100 * g[1]:+.1f}] pts |")
open(f"{OUT}/audiolen.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
log("done")
