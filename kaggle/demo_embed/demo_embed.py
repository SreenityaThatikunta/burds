"""
Build the demo's reference index from openly licensed media (GPU, runs on Kaggle).

1. Embed every iNaturalist photo (280 image tokens) and every xeno-canto recording with the
   audio recipe that won the SSW60 experiments:
     half speed, loudest 5 s | half speed, loudest 10 s | normal speed, loudest 10 s  (3 x 768)
2. Train one linear classifier per modality on this CC data only (C chosen by 5-fold CV) and pick
   the confidence-fusion gamma on synthetic photo+song pairs built from out-of-fold predictions.
3. Honest check: score that web-trained model on the SSW60 *test* sets it has never seen
   (embeddings from earlier runs, same recipe), photo, song and photo + song.

Outputs (/kaggle/working): demo_index.npz (weights + embeddings, a few MB), credits.csv,
demo_eval.json, demo_eval.md
"""
import csv
import glob
import io
import json
import os
import pickle
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

T0 = time.time()
OUT, WORK = "/kaggle/working", "/kaggle/temp"
SR = 16000
BUDGET = 280
AUDIO_KINDS = ("slow2_trim5", "slow2_trim10", "trim10")
N_AUG = 2  # extra noisy copies of every recording (training only): field background from another clip
WORKER = os.environ.get("BURDS_WORKER")


def log(m):
    tag = f"gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '-')}" if WORKER else "main"
    print(f"[{time.time() - T0:7.1f}s {tag}] {m}", flush=True)


# ============================================================================ audio (same as app)
def load_audio(path):
    import soundfile as sf
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001  (older libsndfile without mp3)
        import librosa
        x, sr = librosa.load(path, sr=None, mono=True)
    if x.ndim > 1:
        x = x.mean(1)
    return np.asarray(x, np.float32), sr


def resample(x, sr_in, sr_out):
    from math import gcd
    from scipy.signal import resample_poly
    sr_in, sr_out = int(round(sr_in)), int(round(sr_out))
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def loudest(x, sr, secs):
    from scipy.signal import butter, sosfilt
    n = int(secs * sr)
    if len(x) <= n:
        return 0
    hp = sosfilt(butter(4, 1000, "highpass", fs=sr, output="sos"), x)
    c = np.concatenate([[0.0], np.cumsum(hp.astype(np.float64) ** 2)])
    return int(np.argmax(c[n:] - c[:-n]))


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)


def add_background(x, sr, noise, sr_n, snr_db):
    """Mix in the quietest stretch of another recording (wind, traffic, other birds, hiss) at snr_db."""
    if sr_n != sr:
        noise = resample(noise, sr_n, sr)
    n = min(len(x), len(noise))
    c = np.concatenate([[0.0], np.cumsum(noise.astype(np.float64) ** 2)])
    s0 = int(np.argmin(c[n:] - c[:-n])) if len(noise) > n else 0
    seg = np.resize(noise[s0:s0 + n], len(x))  # tile if the background is shorter than the clip
    return (x + seg * (rms(x) / rms(seg)) * 10 ** (-snr_db / 20)).astype(np.float32)


def audio_variant(x, sr, kind):
    """kind: [slow2_]trim<N>. Half speed reads the samples at sr/2, so frequency f lands at f/2."""
    rate = sr / 2 if kind.startswith("slow2") else sr
    n = float(kind.split("trim")[1])
    s = loudest(x, sr, n)
    return resample(x[s:s + int(n * sr)], rate, SR)


# ============================================================================ GPU worker
def worker(shard, nshards):
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    items = json.load(open(f"{WORK}/items.json"))
    mine = [i for i in range(len(items)) if i % nshards == shard]
    audio_ids = [i for i, it in enumerate(items) if it["kind"] == "audio"]
    cfgs = glob.glob("/kaggle/input/**/embeddinggemma*/**/config.json", recursive=True)
    path = os.path.dirname(sorted(cfgs, key=len)[0]) if cfgs else "google/embeddinggemma-2"
    model = SentenceTransformer(path, device="cuda", model_kwargs={"torch_dtype": torch.float32})
    log(f"model loaded on {torch.cuda.get_device_name(0)}; {len(mine)} items")

    def prep(i):
        it = items[i]
        p = f"{WORK}/{it['file']}"
        try:
            if it["kind"] == "image":
                im = Image.open(p); im.draft("RGB", (1024, 1024))  # noqa: E702
                im = im.convert("RGB"); im.thumbnail((1024, 1024))  # noqa: E702
                return [{"image": im}]
            x, sr = load_audio(p)
            if len(x) < sr:  # < 1 s: useless
                return None
            out = [{"audio": audio_variant(x, sr, k)} for k in AUDIO_KINDS]
            rng_i = np.random.default_rng(i)
            for _ in range(N_AUG):
                j = int(rng_i.choice(audio_ids))
                xn, srn = load_audio(f"{WORK}/{items[j]['file']}") if j != i else (x[::-1].copy(), sr)
                xa = add_background(x, sr, xn, srn, rng_i.uniform(0, 12))
                out += [{"audio": audio_variant(xa, sr, k)} for k in AUDIO_KINDS]
            return out
        except Exception as e:  # noqa: BLE001
            log(f"skip {it['file']}: {e}")
            return None

    pool = ThreadPoolExecutor(8)
    res = {}
    for kind in ("image", "audio"):
        ids = [i for i in mine if items[i]["kind"] == kind]
        kw = {"processing_kwargs": {"image": {"max_soft_tokens": BUDGET}}} if kind == "image" else {}
        for c0 in range(0, len(ids), 32):
            chunk = ids[c0:c0 + 32]
            ins = list(pool.map(prep, chunk))
            ok = [(i, x) for i, x in zip(chunk, ins) if x]
            flat = [d for _, x in ok for d in x]
            if not flat:
                continue
            e = model.encode(flat, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, **kw)
            per = len(AUDIO_KINDS) * (1 + N_AUG) if kind == "audio" else 1
            for j, (i, _) in enumerate(ok):
                # row 0 = clean recording, rows 1.. = noisy copies (audio only)
                res[i] = e[j * per:(j + 1) * per].reshape(1 if kind == "image" else 1 + N_AUG, -1).astype(np.float32)
            if c0 % (320 if kind == "image" else 96) == 0:
                log(f"{kind}: {c0 + len(chunk)}/{len(ids)}")
    pickle.dump(res, open(f"{WORK}/part{shard}.pkl", "wb"))
    sys.exit(0)


if WORKER:
    worker(*map(int, WORKER.split("/")))


# ============================================================================ main
def sh(cmd):
    log(f"$ {cmd}")
    subprocess.run(["bash", "-c", "set -euo pipefail; " + cmd], check=True)


sh("pip install -q -U sentence-transformers transformers soundfile")
import torch  # noqa: E402

os.makedirs(WORK, exist_ok=True)
# self-test of the augmentation before any GPU time is spent
_x = np.sin(np.arange(44100 * 4) / 7).astype(np.float32)
_n = np.random.default_rng(0).normal(size=22050 * 9).astype(np.float32)
for _snr in (0, 6, 12):
    _m = add_background(_x, 44100, _n, 22050, _snr)
    assert _m.shape == _x.shape and np.isfinite(_m).all()
    assert abs(20 * np.log10(rms(_x) / rms(_m - _x)) - _snr) < 0.5, "SNR off"
assert all(audio_variant(_x, 44100, k).ndim == 1 for k in AUDIO_KINDS)
log("augmentation self-test ok")
for t in glob.glob("/kaggle/input/**/images*.tar", recursive=True) + glob.glob("/kaggle/input/**/audio*.tar", recursive=True):
    if "ssw60" in t or "audio_ml" in t:
        continue
    log(f"untar {t}")
    tarfile.open(t).extractall(WORK)
man = []
for m in sorted(glob.glob("/kaggle/input/**/manifest*.csv", recursive=True)):
    man += list(csv.DictReader(open(m)))
man = list({r["file"]: r for r in man if os.path.exists(f"{WORK}/{r['file']}")}.values())  # dedupe
taxa = list(csv.DictReader(open(glob.glob("/kaggle/input/**/burds-demo-data/taxa.csv", recursive=True)[0])))
log(f"{len(man)} files: {sum(r['kind'] == 'image' for r in man)} photos, {sum(r['kind'] == 'audio' for r in man)} songs")
json.dump(man, open(f"{WORK}/items.json", "w"))

ngpu = max(1, torch.cuda.device_count())
procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__)],
                          env={**os.environ, "BURDS_WORKER": f"{s}/{ngpu}", "CUDA_VISIBLE_DEVICES": str(s)})
         for s in range(ngpu)]
assert all(p.wait() == 0 for p in procs), "a GPU worker failed"
emb = {}
for s in range(ngpu):
    emb.update(pickle.load(open(f"{WORK}/part{s}.pkl", "rb")))
log(f"embedded {len(emb)}/{len(man)}")
pickle.dump({"emb": emb, "man": man}, open(f"{OUT}/emb_all.pkl", "wb"))  # lets training be re-run on CPU

# ---------------------------------------------------------------- train on CC data only
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

labels = np.array([int(t["label"]) for t in taxa])
lab2idx = {int(l): i for i, l in enumerate(labels)}
K = len(labels)
GRID = (3, 10, 30, 100, 300)
CV = StratifiedKFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)


def acc(P, y, k=1):
    return float(np.mean((np.argsort(-P, 1)[:, :k] == y[:, None]).any(1)))


def norm_blocks(X, d=768):
    X = X.reshape(len(X), -1, d)
    return (X / np.linalg.norm(X, axis=2, keepdims=True).clip(1e-12)).reshape(len(X), -1)


def fit(X, y):
    lr = lambda C: LogisticRegression(C=C, max_iter=4000, n_jobs=-1)  # noqa: E731
    oof = {C: cross_val_predict(lr(C), X, y, cv=CV, method="predict_proba") for C in GRID}
    C = max(GRID, key=lambda c: acc(oof[c], y))
    return lr(C).fit(X, y), oof[C], C


def fuse_conf(Pi, Pa, g):
    a, b = Pi.max(1, keepdims=True) ** g, Pa.max(1, keepdims=True) ** g
    return (a * np.log(Pi + 1e-9) + b * np.log(Pa + 1e-9)) / (a + b)


def pairs(yi, ya, n=25):
    out = []
    for c in range(K):
        ii, aa = np.where(yi == c)[0], np.where(ya == c)[0]
        if len(ii) and len(aa):
            out += [(rng.choice(ii), rng.choice(aa), c) for _ in range(n)]
    return np.array(out)


data = {}
ix = [i for i in sorted(emb) if man[i]["kind"] == "image"]
X = norm_blocks(np.stack([emb[i][0] for i in ix]))
y = np.array([lab2idx[int(man[i]["label"])] for i in ix])
clf, oof, C = fit(X, y)
data["image"] = {"ix": ix, "X": X, "y": y, "clf": clf, "oof": oof, "C": C}
log(f"image: n={len(y)}  5-fold CV top-1 {acc(oof, y):.3f}  (C={C})")

# Songs: three training recipes, compared fairly. One fold assignment over ALL clean recordings
# (xeno-canto + iNaturalist); each recipe trains on its own rows of the training folds (noisy
# copies stay with their source recording, so nothing leaks) and is scored on every clean
# held-out recording. The shipped recipe is chosen by this CV, not by the SSW60 test.
from joblib import Parallel, delayed  # noqa: E402

aix = [i for i in sorted(emb) if man[i]["kind"] == "audio"]
src = np.array(["inat" if "/inat_" in man[i]["file"] else "xc" for i in aix])
ya = np.array([lab2idx[int(man[i]["label"])] for i in aix])
Xc = norm_blocks(np.stack([emb[i][0] for i in aix]))
Xn = [norm_blocks(np.stack([emb[i][a] for i in aix])) for a in range(1, 1 + N_AUG)]
RECIPES = {"xeno-canto": (("xc",), False), "xeno-canto + iNaturalist": (("xc", "inat"), False),
           "xeno-canto + iNaturalist + noise": (("xc", "inat"), True)}
AGRID = (100, 300, 1000)
folds = list(CV.split(Xc, ya))


def train_rows(rows, recipe):
    srcs, noisy = RECIPES[recipe]
    rows = rows[np.isin(src[rows], srcs)]
    X = np.vstack([Xc[rows]] + ([N[rows] for N in Xn] if noisy else []))
    return X, np.concatenate([ya[rows]] * (1 + (N_AUG if noisy else 0)))


def fold_fit(recipe, C, tr, te):
    X, y_ = train_rows(tr, recipe)
    m = LogisticRegression(C=C, max_iter=4000).fit(X, y_)
    P = np.zeros((len(te), K)); P[:, m.classes_] = m.predict_proba(Xc[te])  # noqa: E702
    return recipe, C, te, P


jobs = [(r, C, tr, te) for r in RECIPES for C in AGRID for tr, te in folds]
oofs = {}
for r, C, te, P in Parallel(n_jobs=-1)(delayed(fold_fit)(*j) for j in jobs):
    oofs.setdefault((r, C), np.zeros((len(ya), K)))[te] = P
audio_models = {}
for r in RECIPES:
    C = max(AGRID, key=lambda c: acc(oofs[(r, c)], ya))
    X, y_ = train_rows(np.arange(len(ya)), r)
    audio_models[r] = {"clf": LogisticRegression(C=C, max_iter=4000).fit(X, y_), "oof": oofs[(r, C)], "C": C,
                       "cv_all": acc(oofs[(r, C)], ya), "cv_xc": acc(oofs[(r, C)][src == "xc"], ya[src == "xc"]),
                       "cv_inat": acc(oofs[(r, C)][src == "inat"], ya[src == "inat"]), "n_train": int(len(y_))}
    log(f"song recipe {r}: CV all {audio_models[r]['cv_all']:.3f}  xc {audio_models[r]['cv_xc']:.3f}  "
        f"inat {audio_models[r]['cv_inat']:.3f}  (C={C}, {len(y_)} training rows)")
best_recipe = max(RECIPES, key=lambda r: audio_models[r]["cv_all"])
log(f"shipping song recipe: {best_recipe}")
am = audio_models[best_recipe]
data["audio"] = {"ix": aix, "X": Xc, "y": ya, "clf": am["clf"], "oof": am["oof"], "C": am["C"]}

pr = pairs(data["image"]["y"], data["audio"]["y"])
gamma = max([0.5, 1, 2, 4, 8], key=lambda g: acc(fuse_conf(data["image"]["oof"][pr[:, 0]],
                                                           data["audio"]["oof"][pr[:, 1]], g), pr[:, 2]))
cv_fused = acc(fuse_conf(data["image"]["oof"][pr[:, 0]], data["audio"]["oof"][pr[:, 1]], gamma), pr[:, 2])
log(f"gamma={gamma}  CV fused top-1 {cv_fused:.3f}")

# ---------------------------------------------------------------- honest test on SSW60 (never seen)
R = {"cc_train": {k: {"n": int(len(v["y"])), "cv_top1": acc(v["oof"], v["y"]), "C": v["C"]} for k, v in data.items()},
     "gamma": gamma, "cv_fused_top1": cv_fused}
try:
    B = np.load(glob.glob("/kaggle/input/**/embeddings.npz", recursive=True)[0], allow_pickle=True)
    A = np.load(glob.glob("/kaggle/input/**/embed_audiolen.npz", recursive=True)[0], allow_pickle=True)
    V = np.load(glob.glob("/kaggle/input/**/embed_v3.npz", recursive=True)[0], allow_pickle=True)
    yidx = lambda yy: np.array([lab2idx[int(v)] for v in yy])  # noqa: E731
    ite = B["img_split"] == "test"
    Pi_all = data["image"]["clf"].predict_proba(norm_blocks(np.asarray(B["img"], np.float32)))
    # SSW60 clips are 10 s, so "loudest 10 s" is the whole clip (aud_slow2 / aud_base).
    Xa = norm_blocks(np.hstack([A["aud_slow2_trim5"], V["aud_slow2"], V["aud_base"]]).astype(np.float32))
    ate = A["aud_split"] == "test"
    Pa_all = data["audio"]["clf"].predict_proba(Xa)
    R["song_recipes"] = {r: {k: v for k, v in m.items() if k not in ("clf", "oof")} |
                         {"ssw60_song_top1": acc(m["clf"].predict_proba(Xa)[ate], yidx(A["aud_y"])[ate]),
                          "ssw60_song_top5": acc(m["clf"].predict_proba(Xa)[ate], yidx(A["aud_y"])[ate], 5)}
                         for r, m in audio_models.items()}
    R["shipped_song_recipe"] = best_recipe
    yi_all, ya_all = yidx(B["img_y"]), yidx(A["aud_y"])
    row_of = {a: i for i, a in enumerate(A["aud_id"].astype(str))}
    b2a = np.array([row_of[a] for a in B["aud_id"].astype(str)])
    PTE = B["pairs_test"]
    Pi_p, Pa_p, y_p = Pi_all[PTE[:, 0]], Pa_all[b2a[PTE[:, 1]]], yidx(PTE[:, 2])
    Pf = fuse_conf(Pi_p, Pa_p, gamma)
    hi, hf = (Pi_p.argmax(1) == y_p).astype(float), (Pf.argmax(1) == y_p).astype(float)
    bidx = rng.integers(0, len(y_p), (1000, len(y_p)))
    d = (hf[bidx] - hi[bidx]).mean(1)
    R["ssw60_test"] = {
        "photo": {"n": int(ite.sum()), "top1": acc(Pi_all[ite], yi_all[ite]), "top5": acc(Pi_all[ite], yi_all[ite], 5)},
        "song": {"n": int(ate.sum()), "top1": acc(Pa_all[ate], ya_all[ate]), "top5": acc(Pa_all[ate], ya_all[ate], 5)},
        "paired": {"n": int(len(y_p)), "photo": acc(Pi_p, y_p), "song": acc(Pa_p, y_p), "fused": acc(Pf, y_p),
                   "gain_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}}
    log(f"SSW60 test: {R['ssw60_test']}")
except Exception as e:  # noqa: BLE001
    log(f"SSW60 check skipped: {e}")

# ---------------------------------------------------------------- export
img_ix = data["image"]["ix"]
np.savez_compressed(
    f"{OUT}/demo_index.npz",
    labels=labels, common=np.array([t["common_name"] for t in taxa]), sci=np.array([t["scientific_name"] for t in taxa]),
    W_img=data["image"]["clf"].coef_.astype(np.float32), b_img=data["image"]["clf"].intercept_.astype(np.float32),
    W_aud=data["audio"]["clf"].coef_.astype(np.float32), b_aud=data["audio"]["clf"].intercept_.astype(np.float32),
    gamma=np.float32(gamma), audio_kinds=np.array(AUDIO_KINDS), budget=np.int32(BUDGET),
    img_emb=data["image"]["X"].astype(np.float16), img_y=data["image"]["y"],
    img_file=np.array([man[i]["file"] for i in img_ix]))
with open(f"{OUT}/credits.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(man[0]))
    w.writeheader(); w.writerows([man[i] for i in sorted(emb) if man[i]["kind"] == "image" or
                                  RECIPES[best_recipe][0].count("inat" if "/inat_" in man[i]["file"] else "xc")])  # noqa: E702
# one display photo per species: the most typical one (closest to the species' mean embedding).
# "Most-faved first" picked e.g. a flight shot for Ruby-throated Hummingbird that looked like a swift.
os.makedirs(f"{OUT}/thumbs", exist_ok=True)
from PIL import Image  # noqa: E402
Xi, yi_ = data["image"]["X"], data["image"]["y"]
for c in range(K):
    rows_c = np.where(yi_ == c)[0]
    mu = Xi[rows_c].mean(0)
    i = img_ix[rows_c[int(np.argmax(Xi[rows_c] @ mu))]]
    im = Image.open(f"{WORK}/{man[i]['file']}").convert("RGB"); im.thumbnail((320, 320))  # noqa: E702
    im.save(f"{OUT}/thumbs/{taxa[c]['species_code']}.jpg", quality=85)
json.dump(R, open(f"{OUT}/demo_eval.json", "w"), indent=2)

pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
md = ["# Demo model: trained only on openly licensed web data", "",
      f"Training data: {R['cc_train']['image']['n']} iNaturalist photos, {int((src == 'xc').sum())} xeno-canto + "
      f"{int((src == 'inat').sum())} iNaturalist recordings (60 species); song recipe: {best_recipe}. 5-fold CV top-1: photo {pct(R['cc_train']['image']['cv_top1'])}, "
      f"song {pct(R['cc_train']['audio']['cv_top1'])}, photo + song {pct(cv_fused)} (gamma={gamma}).", ""]
if "ssw60_test" in R:
    s = R["ssw60_test"]
    g = s["paired"]["gain_ci95"]
    md += ["## Tested on SSW60 (different photographers, recordists and sites; never seen)", "",
           "| input | n | top-1 | top-5 |", "|---|---|---|---|",
           f"| photo | {s['photo']['n']} | {pct(s['photo']['top1'])} | {pct(s['photo']['top5'])} |",
           f"| song | {s['song']['n']} | {pct(s['song']['top1'])} | {pct(s['song']['top5'])} |",
           f"| photo + song (paired) | {s['paired']['n']} | **{pct(s['paired']['fused'])}** | |", "",
           f"Paired set: photo only {pct(s['paired']['photo'])}, gain from adding song "
           f"[{100 * g[0]:+.1f}, {100 * g[1]:+.1f}] pts (95% CI)."]
if "song_recipes" in R:
    md += ["", "## Song training recipes (chosen by CV on the web data; SSW60 shown for reference)", "",
           "| recipe | training rows | CV, all clean recordings | CV, iNaturalist only | SSW60 song top-1 | top-5 |",
           "|---|---|---|---|---|---|"]
    for r, m in R["song_recipes"].items():
        md.append(f"| {'**' + r + '**' if r == best_recipe else r} | {m['n_train']} | {pct(m['cv_all'])} | "
                  f"{pct(m['cv_inat'])} | {pct(m['ssw60_song_top1'])} | {pct(m['ssw60_song_top5'])} |")
open(f"{OUT}/demo_eval.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
log("done")
