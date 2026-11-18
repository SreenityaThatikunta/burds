"""
Evaluate the step-B audio variants and step-C photo token budgets (CPU only).

Inputs (kernel_sources):
  burds-ssw60-embed     -> embed_v3.npz, timing.json   (new variants)
  burds-ssw60-baseline  -> embeddings.npz, taxa.csv    (full photo set + paired test set)
Protocol matches ssw60_analysis: settings chosen by 5-fold CV on train only, test used once,
95% bootstrap CIs.
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


V = np.load(find("embed_v3.npz"), allow_pickle=True)
B = np.load(find("embeddings.npz"), allow_pickle=True)
timing = json.load(open(find("timing.json")))
L = B["labels"]
K = len(L)
lab2idx = {int(l): i for i, l in enumerate(L)}
yidx = lambda y: np.array([lab2idx[int(v)] for v in y])  # noqa: E731


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
    """CV-tuned logistic regression -> (train OOF probs, test probs, C)."""
    scores = {C: cross_val_predict(lr(C), Xtr, ytr, cv=CV, method="predict_proba") for C in GRID}
    C = max(GRID, key=lambda c: acc(scores[c], ytr))
    return scores[C], lr(C).fit(Xtr, ytr).predict_proba(Xte), C


def centroid_acc(Xtr, ytr, Xte, yte):
    Cn = norm(np.stack([Xtr[ytr == k].mean(0) for k in range(K)]))
    return acc(Xte @ Cn.T, yte)


R = {"timing": timing, "image_budget": {}, "audio": {}, "fusion": {}}

# ------------------------------------------------------------------ C: photo token budgets
ys, ss = yidx(V["imgsub_y"]), V["imgsub_split"]
tr, te = ss == "train", ss == "test"
for b in (70, 140, 280):
    k = f"img_b{b}"
    if k not in V:
        continue
    X = norm(V[k])
    _, Pte, C = probe(X[tr], ys[tr], X[te])
    t = timing["per_job"].get(k, {})
    R["image_budget"][b] = {"probe_top1": acc(Pte, ys[te]), "ci95": ci(Pte, ys[te]),
                            "centroid_top1": centroid_acc(X[tr], ys[tr], X[te], ys[te]),
                            "items_per_s_per_gpu": t.get("items_per_s_per_gpu"),
                            "load_wait_s": t.get("load_wait_s"), "encode_s": t.get("encode_s"),
                            "n_train": int(tr.sum()), "n_test": int(te.sum()), "C": C}
    log(f"budget {b}: {R['image_budget'][b]}")

# ------------------------------------------------------------------ B: audio variants
ya, sa = yidx(V["aud_y"]), V["aud_split"]
atr, ate = sa == "train", sa == "test"
n_aud = len(ya)


def clip_level(name):
    E, own = np.asarray(V[name], np.float32), V[name + "__own"]
    M = np.zeros((n_aud, E.shape[1]), np.float32)
    np.add.at(M, own, E)
    return norm(M)


feats = {v: clip_level(f"aud_{v}") for v in ("base", "trim3", "slow2", "slow2_trim3", "win3")}
feats["base+slow2"] = np.hstack([feats["base"], feats["slow2"]])
feats["base+slow2+trim3"] = np.hstack([feats["base"], feats["slow2"], feats["trim3"]])
feats["all"] = np.hstack([feats[v] for v in ("base", "trim3", "slow2", "slow2_trim3", "win3")])
names_human = {"base": "full clip (run 2 setup)", "trim3": "loudest 3 s", "slow2": "half speed",
               "slow2_trim3": "loudest 3 s, half speed", "win3": "mean of 3 s windows",
               "base+slow2": "full + half speed (concatenated)",
               "base+slow2+trim3": "full + half speed + loudest 3 s", "all": "all five concatenated"}
aud_probs = {}
for v, X in feats.items():
    oof, Pte, C = probe(X[atr], ya[atr], X[ate])
    aud_probs[v] = (oof, Pte)
    R["audio"][v] = {"desc": names_human[v], "probe_top1": acc(Pte, ya[ate]), "probe_top5": acc(Pte, ya[ate], 5),
                     "ci95": ci(Pte, ya[ate]), "C": C,
                     "centroid_top1": centroid_acc(X[atr], ya[atr], X[ate], ya[ate]) if X.shape[1] == 768 else None}
    log(f"audio {v}: probe {R['audio'][v]['probe_top1']:.3f}")

# Window voting: probe trained on individual windows, test clip = mean log-prob of its windows.
Ew, ow = norm(V["aud_win3"]), V["aud_win3__own"]
wtr, wte = atr[ow], ate[ow]
_, Pw, Cw = probe(Ew[wtr], ya[ow][wtr], Ew[wte])
logp = np.zeros((n_aud, K))
cnt = np.zeros(n_aud)
np.add.at(logp, ow[wte], np.log(Pw + 1e-9))
np.add.at(cnt, ow[wte], 1)
Pvote = logp[ate] / cnt[ate, None]
R["audio"]["win3_vote"] = {"desc": "3 s windows, probe per window, vote", "probe_top1": acc(Pvote, ya[ate]),
                           "probe_top5": acc(Pvote, ya[ate], 5), "ci95": ci(Pvote, ya[ate]), "C": Cw}
log(f"audio win3_vote: {R['audio']['win3_vote']['probe_top1']:.3f}")

# ------------------------------------------------------------------ fusion with photos (paired test from run 2)
best_v = max((v for v in aud_probs), key=lambda v: R["audio"][v]["probe_top1"])
log(f"best audio variant: {best_v}")
img = norm(B["img"])
yi = yidx(B["img_y"])
itr, ite = B["img_split"] == "train", B["img_split"] == "test"
i_oof, i_te, Ci = probe(img[itr], yi[itr], img[ite])
Fi = np.zeros((len(img), K)); Fi[np.where(itr)[0]] = i_oof; Fi[np.where(ite)[0]] = i_te  # noqa: E702

# Map run-2 audio rows to this run's rows by asset id.
row_of = {a: i for i, a in enumerate(V["aud_id"].astype(str))}
b2v = np.array([row_of[a] for a in B["aud_id"].astype(str)])


def fuse_conf(Pi, Pa, g):
    ci_, ca = Pi.max(1, keepdims=True) ** g, Pa.max(1, keepdims=True) ** g
    return (ci_ * np.log(Pi + 1e-9) + ca * np.log(Pa + 1e-9)) / (ci_ + ca)


def fuse_w(Pi, Pa, w):
    return w * np.log(Pi + 1e-9) + (1 - w) * np.log(Pa + 1e-9)


PTR, PTE = B["pairs_train"], B["pairs_test"]
ytr_p, yte_p = yidx(PTR[:, 2]), yidx(PTE[:, 2])
for v in ("base", best_v):
    Fa = np.zeros((n_aud, K)); Fa[np.where(atr)[0]] = aud_probs[v][0]; Fa[np.where(ate)[0]] = aud_probs[v][1]  # noqa: E702
    Pi_tr, Pa_tr = Fi[PTR[:, 0]], Fa[b2v[PTR[:, 1]]]
    Pi_te, Pa_te = Fi[PTE[:, 0]], Fa[b2v[PTE[:, 1]]]
    g = max([0.5, 1, 2, 4, 8], key=lambda g: acc(fuse_conf(Pi_tr, Pa_tr, g), ytr_p))
    w = max(np.linspace(0, 1, 21), key=lambda w: acc(fuse_w(Pi_tr, Pa_tr, w), ytr_p))
    rows = {"image_only": Pi_te, "audio_only": Pa_te, "weighted": fuse_w(Pi_te, Pa_te, w),
            "confidence": fuse_conf(Pi_te, Pa_te, g)}
    R["fusion"][v] = {k: {"top1": acc(P, yte_p), "ci95": ci(P, yte_p)} for k, P in rows.items()}
    R["fusion"][v]["params"] = {"gamma": g, "image_weight": float(w), "image_C": Ci}
    # Paired difference vs image-only (same test pairs): bootstrap CI of the gain.
    hit_i = (Pi_te.argmax(1) == yte_p).astype(float)
    hit_f = (rows["confidence"].argmax(1) == yte_p).astype(float)
    bidx = rng.integers(0, len(yte_p), (1000, len(yte_p)))
    d = (hit_f[bidx] - hit_i[bidx]).mean(1)
    R["fusion"][v]["confidence_gain_vs_image_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    log(f"fusion with {v}: " + "  ".join(f"{k}={x['top1']:.3f}" for k, x in R['fusion'][v].items() if 'top1' in x))

# ------------------------------------------------------------------ write
R = json.loads(json.dumps(R, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
json.dump(R, open(f"{OUT}/variants.json", "w"), indent=2)
pct = lambda x: "–" if x is None else f"{100 * x:.1f}%"  # noqa: E731
cis = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"  # noqa: E731
md = ["# Step B + C: audio variants and photo token budgets", "",
      f"Embedding wall time: **{timing['wall_min']} min** on {timing['gpus']} GPU(s).", "",
      "## Photo token budget (40 photos per species subset; linear probe)", "",
      "| tokens/photo | top-1 | 95% CI | centroid | photos/s per GPU | load wait | encode |",
      "|---|---|---|---|---|---|---|"]
for b, r in R["image_budget"].items():
    md.append(f"| {b} | {pct(r['probe_top1'])} | {cis(r['ci95'])} | {pct(r['centroid_top1'])} | "
              f"{r['items_per_s_per_gpu']} | {r['load_wait_s']}s | {r['encode_s']}s |")
md += ["", "## Audio variants (full audio test set, n=%d; linear probe)" % int(ate.sum()), "",
       "| variant | top-1 | 95% CI | top-5 | centroid |", "|---|---|---|---|---|"]
for v, r in sorted(R["audio"].items(), key=lambda kv: -kv[1]["probe_top1"]):
    md.append(f"| {r['desc']} | {pct(r['probe_top1'])} | {cis(r['ci95'])} | {pct(r['probe_top5'])} | "
              f"{pct(r.get('centroid_top1'))} |")
md += ["", "## Audio throughput", "", "| job | clips/s per GPU | inputs | load wait | encode |", "|---|---|---|---|---|"]
for k, t in timing["per_job"].items():
    if k.startswith("aud_"):
        md.append(f"| {k} | {t['items_per_s_per_gpu']} | {t['inputs']} | {t['load_wait_s']}s | {t['encode_s']}s |")
md += ["", f"## Photo + song fusion (paired test, n={len(yte_p)})", "",
       "| audio features | image only | audio only | weighted | confidence | gain vs image (95% CI) |",
       "|---|---|---|---|---|---|"]
for v, r in R["fusion"].items():
    g = r["confidence_gain_vs_image_ci95"]
    md.append(f"| {names_human.get(v, v)} | {pct(r['image_only']['top1'])} | {pct(r['audio_only']['top1'])} | "
              f"{pct(r['weighted']['top1'])} | {pct(r['confidence']['top1'])} | "
              f"[{100 * g[0]:+.1f}, {100 * g[1]:+.1f}] pts |")
if timing.get("notes"):
    md += ["", "Notes: " + "; ".join(timing["notes"])]
open(f"{OUT}/variants.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
log("done")
