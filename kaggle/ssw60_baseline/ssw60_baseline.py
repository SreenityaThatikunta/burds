"""
SSW60 baseline: zero-training bird identification with EmbeddingGemma 2.

Runs entirely on Kaggle (GPU + internet). Steps:
  0. Install deps, load the model, and smoke-test image/audio/fused encoding (fail fast).
  1. Stream the SSW60 tarball from S3 and extract ONLY metadata, audio_ml/ and
     images_nabirds/ (video is skipped for now), so the 31 GB archive never sits on disk.
  2. Embed train/test images, train/test audio, species names, and image+audio pairs.
  3. Evaluate nearest-centroid and 1-NN classification for each modality, late fusion,
     joint (single-embedding) fusion, text zero-shot, and Matryoshka truncation.
  4. Write results.json / results.md / embeddings.npz to /kaggle/working.

SSW60 media is research-only and must not be redistributed: keep this kernel private and
never copy images/audio out of Kaggle. Only metrics and embeddings leave the kernel.
"""
import glob
import json
import os
import random
import subprocess
import sys
import time

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def sh(cmd):
    log(f"$ {cmd}")
    subprocess.run(["bash", "-c", "set -euo pipefail; " + cmd], check=True)


# --------------------------------------------------------------------------- 0. setup
sh("pip install -q -U sentence-transformers transformers soundfile")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
import transformers  # noqa: E402
import sentence_transformers  # noqa: E402

log(f"python {sys.version.split()[0]} torch {torch.__version__} "
    f"transformers {transformers.__version__} sentence-transformers {sentence_transformers.__version__}")
log(f"cuda: {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
OUT = "/kaggle/working"
DATA = "/tmp/ssw60"
PAIRS_PER_CLASS = 25  # image+audio pairs per species, for both train gallery and test queries
DIMS = [768, 512, 256, 128]

# Prefer the Kaggle-mounted model (model_sources); fall back to the Hugging Face hub.
cfgs = glob.glob("/kaggle/input/**/embeddinggemma*/**/config.json", recursive=True) + \
       glob.glob("/kaggle/input/embeddinggemma-2/**/config.json", recursive=True)
MODEL_PATH = os.path.dirname(sorted(cfgs, key=len)[0]) if cfgs else "google/embeddinggemma-2"
log(f"model path: {MODEL_PATH}")

# bf16 only on Ampere+; T4/P100 emulate it slowly, so use fp32 there (never fp16 per model card).
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
dtype = torch.bfloat16 if use_bf16 else torch.float32
model = SentenceTransformer(MODEL_PATH, device="cuda" if torch.cuda.is_available() else "cpu",
                            model_kwargs={"torch_dtype": dtype})
log(f"loaded model, dtype={dtype}, modalities={getattr(model, 'modalities', '?')}")

AUDIO_SR = 16000  # model expects mono 16 kHz arrays


def encode(items, batch_size=16):
    """Encode a list of inputs (str or modality dicts) -> (N, 768) L2-normalised float32."""
    out = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        try:
            e = model.encode(chunk, batch_size=batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
        except Exception as ex:  # some input mixes may not batch; fall back to one at a time
            if i == 0:
                log(f"batched encode failed ({type(ex).__name__}: {ex}); falling back to per-item")
            e = np.stack([model.encode(x, normalize_embeddings=True, convert_to_numpy=True,
                                       show_progress_bar=False) for x in chunk])
        out.append(np.asarray(e, dtype=np.float32))
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def load_audio(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != AUDIO_SR:
        g = np.gcd(sr, AUDIO_SR)
        x = resample_poly(x, AUDIO_SR // g, sr // g).astype(np.float32)
    return x


def load_image(path):
    return Image.open(path).convert("RGB")


# Smoke test before spending ~30+ min streaming data.
_img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
_aud = (0.1 * np.sin(2 * np.pi * 2000 * np.arange(AUDIO_SR * 2) / AUDIO_SR)).astype(np.float32)
_smoke = encode([{"image": _img}, {"audio": _aud},
                 {"text": "<|image|><|audio|>", "image": _img, "audio": _aud}, "a Canada Goose"])
log(f"smoke test ok: {_smoke.shape}")

# --------------------------------------------------------------------------- 1. data
URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/ssw60/ssw60.tar.gz"
os.makedirs(DATA, exist_ok=True)
if not glob.glob(f"{DATA}/**/taxa.csv", recursive=True):
    dec = "pigz -dc" if subprocess.run("command -v pigz", shell=True).returncode == 0 else "gzip -dc"
    # Stream: nothing of the 31 GB archive is stored; only matching members are written.
    # Only patterns known to exist: GNU tar exits non-zero if any pattern matches nothing.
    sh(f"curl -sSfL --retry 5 '{URL}' | {dec} | tar -x -C {DATA} --wildcards "
       f"'*.csv' '*audio_ml/*' '*images_nabirds/*'")
sh(f"du -sh {DATA}; find {DATA} -maxdepth 2 | head -30")

ROOT = os.path.dirname(glob.glob(f"{DATA}/**/taxa.csv", recursive=True)[0])
taxa = pd.read_csv(f"{ROOT}/taxa.csv")
aud = pd.read_csv(f"{ROOT}/audio_ml.csv")
img = pd.read_csv(f"{ROOT}/images_nabirds.csv")
aud["path"] = [f"{ROOT}/audio_ml/{a}.wav" for a in aud.asset_id]
img["path"] = [f"{ROOT}/images_nabirds/{a}.jpg" for a in img.asset_id]
aud = aud[[os.path.exists(p) for p in aud.path]].reset_index(drop=True)
img = img[[os.path.exists(p) for p in img.path]].reset_index(drop=True)
log(f"species={len(taxa)}  audio={len(aud)} {aud.split.value_counts().to_dict()}  "
    f"images={len(img)} {img.split.value_counts().to_dict()}")
LABELS = sorted(taxa.label.unique())
NAME = dict(zip(taxa.label, taxa.common_name))

# --------------------------------------------------------------------------- 2. embed


def embed_images(paths, chunk=128):
    out = []
    for i in range(0, len(paths), chunk):
        out.append(encode([{"image": load_image(p)} for p in paths[i:i + chunk]]))
        log(f"  images {min(i + chunk, len(paths))}/{len(paths)}")
    return np.concatenate(out)


def embed_audio(paths, chunk=64):
    out = []
    for i in range(0, len(paths), chunk):
        out.append(encode([{"audio": load_audio(p)} for p in paths[i:i + chunk]], batch_size=8))
        log(f"  audio {min(i + chunk, len(paths))}/{len(paths)}")
    return np.concatenate(out)


def make_pairs(split):
    """Pair each sampled image with a random same-species audio clip (SSW60 media is unpaired)."""
    rng = np.random.default_rng(SEED + (split == "test"))
    pairs = []
    for lab in LABELS:
        ii = img.index[(img.label == lab) & (img.split == split)].to_numpy()
        aa = aud.index[(aud.label == lab) & (aud.split == split)].to_numpy()
        if len(ii) == 0 or len(aa) == 0:
            continue
        sel_i = rng.choice(ii, size=min(PAIRS_PER_CLASS, len(ii)), replace=False)
        sel_a = rng.choice(aa, size=len(sel_i), replace=len(aa) < len(sel_i))
        pairs += [(int(i), int(a), int(lab)) for i, a in zip(sel_i, sel_a)]
    return pairs


def embed_pairs(pairs, chunk=32):
    out = []
    for i in range(0, len(pairs), chunk):
        items = [{"text": "<|image|><|audio|>", "image": load_image(img.path[a]),
                  "audio": load_audio(aud.path[b])} for a, b, _ in pairs[i:i + chunk]]
        out.append(encode(items, batch_size=8))
        log(f"  pairs {min(i + chunk, len(pairs))}/{len(pairs)}")
    return np.concatenate(out)


t = time.time()
log("embedding images")
E_img = embed_images(list(img.path))
log("embedding audio")
E_aud = embed_audio(list(aud.path))
log("embedding species names")
E_txt = encode([f"{NAME[l]}, a bird species" for l in LABELS])
pairs = {s: make_pairs(s) for s in ("train", "test")}
log(f"embedding fused pairs train={len(pairs['train'])} test={len(pairs['test'])}")
E_pair = {s: embed_pairs(pairs[s]) for s in ("train", "test")}
embed_secs = time.time() - t

# --------------------------------------------------------------------------- 3. evaluate


def trunc(E, d):
    E = E[:, :d]
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-12)


def centroids(E, y):
    C = np.stack([E[y == l].mean(0) for l in LABELS])
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def topk(scores, y, k):
    top = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean([y[i] in np.asarray(LABELS)[top[i]] for i in range(len(y))]))


def evaluate(Etr, ytr, Ete, yte):
    S = Ete @ centroids(Etr, ytr).T
    nn = ytr[np.argmax(Ete @ Etr.T, axis=1)]
    return {"centroid_top1": topk(S, yte, 1), "centroid_top5": topk(S, yte, 5),
            "nn1_top1": float(np.mean(nn == yte)), "n_test": int(len(yte))}


results = {"meta": {"model": MODEL_PATH, "dtype": str(dtype), "n_species": len(LABELS),
                    "pairs_per_class": PAIRS_PER_CLASS, "embed_seconds": round(embed_secs, 1),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                 "sentence_transformers": sentence_transformers.__version__}},
           "full": {}, "paired": {}, "zero_shot_text": {}}

yi, ya = img.label.to_numpy(), aud.label.to_numpy()
tri, tei = (img.split == "train").to_numpy(), (img.split == "test").to_numpy()
tra, tea = (aud.split == "train").to_numpy(), (aud.split == "test").to_numpy()

for d in DIMS:
    Ei, Ea, Et = trunc(E_img, d), trunc(E_aud, d), trunc(E_txt, d)
    Ep = {s: trunc(E_pair[s], d) for s in E_pair}
    r = results["full"][d] = {}
    r["image"] = evaluate(Ei[tri], yi[tri], Ei[tei], yi[tei])
    r["audio"] = evaluate(Ea[tra], ya[tra], Ea[tea], ya[tea])

    # Same paired test set for every row so the comparison is apples to apples.
    pi = np.array([a for a, _, _ in pairs["test"]])
    pa = np.array([b for _, b, _ in pairs["test"]])
    py = np.array([l for _, _, l in pairs["test"]])
    Ci, Ca = centroids(Ei[tri], yi[tri]), centroids(Ea[tra], ya[tra])
    Si, Sa = Ei[pi] @ Ci.T, Ea[pa] @ Ca.T
    ytr_p = np.array([l for _, _, l in pairs["train"]])
    Sj = Ep["test"] @ centroids(Ep["train"], ytr_p).T
    results["paired"][d] = {
        "image_only": topk(Si, py, 1),
        "audio_only": topk(Sa, py, 1),
        "late_fusion": topk(Si + Sa, py, 1),
        "joint_embedding": topk(Sj, py, 1),
        "joint_plus_late": topk(Si + Sa + Sj, py, 1),
        "n_test_pairs": int(len(py)),
    }
    results["zero_shot_text"][d] = {
        "image_to_name_top1": topk(Ei[tei] @ Et.T, yi[tei], 1),
        "audio_to_name_top1": topk(Ea[tea] @ Et.T, ya[tea], 1),
        "pair_to_name_top1": topk(Ep["test"] @ Et.T, py, 1),
    }
    log(f"d={d}: full={r}  paired={results['paired'][d]}  zs={results['zero_shot_text'][d]}")

# Per-species accuracy (768-d, paired test set) to find the hard cases.
Ei, Ea = trunc(E_img, 768), trunc(E_aud, 768)
Si = Ei[pi] @ centroids(Ei[tri], yi[tri]).T
Sa = Ea[pa] @ centroids(Ea[tra], ya[tra]).T
L = np.asarray(LABELS)
per = []
for l in LABELS:
    m = py == l
    if m.any():
        per.append({"label": int(l), "name": NAME[l], "n": int(m.sum()),
                    "image": float(np.mean(L[np.argmax(Si[m], 1)] == l)),
                    "audio": float(np.mean(L[np.argmax(Sa[m], 1)] == l)),
                    "fused": float(np.mean(L[np.argmax(Si[m] + Sa[m], 1)] == l))})
results["per_species_768"] = sorted(per, key=lambda p: p["fused"])

# --------------------------------------------------------------------------- 4. write
results = json.loads(json.dumps(results, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)

pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
md = ["# SSW60 baseline — EmbeddingGemma 2, no training", "",
      f"{len(LABELS)} species · images: NABirds split · audio: SSW60 audio_ml split · "
      f"{results['meta']['gpu']} · embedding time {embed_secs / 60:.1f} min", "",
      "## Paired test set (top-1, nearest class centroid)", "",
      "| dims | image only | audio only | late fusion | joint embedding | joint + late |",
      "|---|---|---|---|---|---|"]
for d in DIMS:
    p = results["paired"][str(d)]
    md.append(f"| {d} | {pct(p['image_only'])} | {pct(p['audio_only'])} | {pct(p['late_fusion'])} | "
              f"{pct(p['joint_embedding'])} | {pct(p['joint_plus_late'])} |")
md += ["", "## Full single-modality test sets", "",
       "| dims | modality | centroid top-1 | centroid top-5 | 1-NN top-1 | n |", "|---|---|---|---|---|---|"]
for d in DIMS:
    for m in ("image", "audio"):
        r = results["full"][str(d)][m]
        md.append(f"| {d} | {m} | {pct(r['centroid_top1'])} | {pct(r['centroid_top5'])} | "
                  f"{pct(r['nn1_top1'])} | {r['n_test']} |")
md += ["", "## Zero-shot (match against species names, no examples)", "",
       "| dims | image→name | audio→name | pair→name |", "|---|---|---|---|"]
for d in DIMS:
    z = results["zero_shot_text"][str(d)]
    md.append(f"| {d} | {pct(z['image_to_name_top1'])} | {pct(z['audio_to_name_top1'])} | "
              f"{pct(z['pair_to_name_top1'])} |")
md += ["", "## Hardest species (768-d, paired)", "", "| species | image | audio | fused |", "|---|---|---|---|"]
for p in results["per_species_768"][:10]:
    md.append(f"| {p['name']} | {pct(p['image'])} | {pct(p['audio'])} | {pct(p['fused'])} |")
with open(f"{OUT}/results.md", "w") as f:
    f.write("\n".join(md) + "\n")

np.savez_compressed(
    f"{OUT}/embeddings.npz",
    img=E_img.astype(np.float16), img_id=img.asset_id.astype(str).to_numpy(), img_y=yi,
    img_split=img.split.astype(str).to_numpy(),
    aud=E_aud.astype(np.float16), aud_id=aud.asset_id.astype(str).to_numpy(), aud_y=ya,
    aud_split=aud.split.astype(str).to_numpy(),
    txt=E_txt.astype(np.float16), labels=np.asarray(LABELS),
    pair_train=E_pair["train"].astype(np.float16), pair_test=E_pair["test"].astype(np.float16),
    pairs_train=np.asarray(pairs["train"]), pairs_test=np.asarray(pairs["test"]))
taxa.to_csv(f"{OUT}/taxa.csv", index=False)  # species metadata only, no media
print("\n".join(md))
log("done")
