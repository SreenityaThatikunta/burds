"""
Fast SSW60 embedding with EmbeddingGemma 2 (steps C + B).

C (speed):
  - Reads the prepped photo/audio tars from the burds-ssw60-data kernel (no 33 GB download).
  - One worker process per GPU (Kaggle "T4" = 2x T4), each owning a shard of the jobs.
  - Photos are decoded in a thread pool with JPEG draft mode (decode at reduced size) and
    prefetched while the GPU encodes the previous chunk. Load vs encode time is logged.
  - Photo token-budget sweep (70 / 140 / 280 soft tokens) on a 40-per-species subset.
B (audio):
  - base        : full 10 s clip at 16 kHz (same as run 2)
  - trim3       : loudest 3 s (energy above 1 kHz) at 16 kHz
  - slow2       : played at half speed, so 0-11 kHz content lands in the model's 0-5.5 kHz range
  - slow2_trim3 : loudest 3 s, then half speed (6 s)
  - win3        : every 3 s window (hop 2.5 s), aggregated later (mean / best window)
Output: /kaggle/working/embed_v3.npz (+ timing.json). Evaluation runs separately on CPU.
SSW60 media is research-only: kernel stays private; only embeddings and metrics leave it.
"""
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

T0 = time.time()
DATA = "/tmp/ssw60"
WORK = "/tmp/emb"
OUT = "/kaggle/working"
SR = 16000
IMG_PER_CLASS = 20  # per split, for the token-budget sweep
BUDGETS = [70, 140, 280]
WORKER = "--worker" in sys.argv


def log(m):
    tag = f"gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '-')}" if WORKER else "main"
    print(f"[{time.time() - T0:7.1f}s {tag}] {m}", flush=True)


# ============================================================================ audio variants
def load_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    return x, sr


def resample(x, sr_in, sr_out):
    from scipy.signal import resample_poly
    from math import gcd
    sr_in, sr_out = int(round(sr_in)), int(round(sr_out))
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def loudest(x, sr, secs=3.0):
    """Start index of the window with most energy above 1 kHz (ignores wind/rumble)."""
    from scipy.signal import butter, sosfilt
    n = int(secs * sr)
    if len(x) <= n:
        return 0
    hp = sosfilt(butter(4, 1000, "highpass", fs=sr, output="sos"), x)
    # Sliding-window energy via cumulative sum: O(len) instead of np.convolve's O(len * n),
    # which took ~10 s per clip for a 3 s window and stalled run v1 of this kernel.
    c = np.concatenate([[0.0], np.cumsum(hp.astype(np.float64) ** 2)])
    return int(np.argmax(c[n:] - c[:-n]))


def audio_variant(path, kind):
    x, sr = load_wav(path)
    if kind == "base":
        return [resample(x, sr, SR)]
    if kind == "trim3":
        s = loudest(x, sr)
        return [resample(x[s:s + 3 * sr], sr, SR)]
    if kind == "slow2":
        return [resample(x, sr / 2, SR)]  # same samples read at half rate = half speed
    if kind == "slow2_trim3":
        s = loudest(x, sr)
        return [resample(x[s:s + 3 * sr], sr / 2, SR)]
    if kind == "win3":
        y = resample(x, sr, SR)
        n, hop = 3 * SR, int(2.5 * SR)
        starts = list(range(0, max(1, len(y) - n + 1), hop)) or [0]
        if len(y) > n and starts[-1] != len(y) - n:
            starts.append(len(y) - n)  # make sure the last 3 s is covered
        return [y[s:s + n] for s in starts]
    raise ValueError(kind)


# ============================================================================ worker
def worker(shard, nshards):
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    jobs = json.load(open(f"{WORK}/jobs.json"))
    cfgs = glob.glob("/kaggle/input/**/embeddinggemma*/**/config.json", recursive=True)
    path = os.path.dirname(sorted(cfgs, key=len)[0]) if cfgs else "google/embeddinggemma-2"
    model = SentenceTransformer(path, device="cuda", model_kwargs={"torch_dtype": torch.float32})
    log(f"model loaded on {torch.cuda.get_device_name(0)}")

    def load_img(p):
        im = Image.open(p)
        im.draft("RGB", (1024, 1024))  # JPEG decodes directly at <=2x the target size
        im = im.convert("RGB")
        im.thumbnail((1024, 1024))
        return im

    pool = ThreadPoolExecutor(8)
    timing = {}
    for job in jobs:
        name, kind, items = job["name"], job["kind"], job["items"]
        mine = [i for i in range(len(items)) if i % nshards == shard]
        t_load = t_enc = 0.0
        embs, owners = [], []
        chunk = 64 if kind == "image" else 32
        loader = (lambda p: {"image": load_img(p)}) if kind == "image" else None

        def load_chunk(ix):
            if kind == "image":
                return list(pool.map(loader, [items[i] for i in ix])), list(ix)
            segs = list(pool.map(lambda i: audio_variant(items[i], job["variant"]), ix))
            ins, own = [], []
            for i, ss in zip(ix, segs):
                ins += [{"audio": s} for s in ss]
                own += [i] * len(ss)
            return ins, own

        chunks = [mine[i:i + chunk] for i in range(0, len(mine), chunk)]
        fut = pool.submit(load_chunk, chunks[0]) if chunks else None
        kw = {"processing_kwargs": {"image": {"max_soft_tokens": job["budget"]}}} if job.get("budget") else {}
        for c in range(len(chunks)):
            t = time.time()
            ins, own = fut.result()
            t_load += time.time() - t
            fut = pool.submit(load_chunk, chunks[c + 1]) if c + 1 < len(chunks) else None
            t = time.time()
            try:
                e = model.encode(ins, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False, **kw)
            except TypeError as ex:  # token budget not supported by this library version
                log(f"{name}: processing_kwargs rejected ({ex}); using default budget")
                kw = {}
                timing.setdefault("_notes", []).append(f"{name}: budget unsupported")
                e = model.encode(ins, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)
            t_enc += time.time() - t
            embs.append(np.asarray(e, np.float32))
            owners += own
            if c % 10 == 0 or c + 1 == len(chunks):
                log(f"{name}: {min((c + 1) * chunk, len(mine))}/{len(mine)}  "
                    f"wait-on-load {t_load:.0f}s  encode {t_enc:.0f}s")
        E = np.concatenate(embs) if embs else np.zeros((0, 768), np.float32)
        np.save(f"{WORK}/{name}.{shard}.emb.npy", E)
        np.save(f"{WORK}/{name}.{shard}.own.npy", np.asarray(owners, np.int64))
        # Also keep a copy in the kernel output so finished jobs survive a later failure.
        os.makedirs(f"{OUT}/parts", exist_ok=True)
        np.save(f"{OUT}/parts/{name}.{shard}.emb.npy", E.astype(np.float16))
        np.save(f"{OUT}/parts/{name}.{shard}.own.npy", np.asarray(owners, np.int64))
        timing[name] = {"items": len(mine), "inputs": int(len(E)), "load_wait_s": round(t_load, 1),
                        "encode_s": round(t_enc, 1)}
    json.dump(timing, open(f"{WORK}/timing.{shard}.json", "w"))
    log("worker done")


if WORKER:
    worker(int(sys.argv[sys.argv.index("--worker") + 1]), int(sys.argv[sys.argv.index("--worker") + 2]))
    sys.exit(0)


# ============================================================================ main
def sh(cmd):
    log(f"$ {cmd}")
    subprocess.run(["bash", "-c", "set -euo pipefail; " + cmd], check=True)


sh("pip install -q -U sentence-transformers transformers soundfile")
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ngpu = torch.cuda.device_count()
log(f"GPUs: {ngpu} x {torch.cuda.get_device_name(0) if ngpu else 'none'}  CPUs: {os.cpu_count()}")

src = os.path.dirname(glob.glob("/kaggle/input/**/images_nabirds.tar", recursive=True)[0])
os.makedirs(DATA, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
sh(f"cp {src}/*.csv {DATA}/ && tar -xf {src}/audio_ml.tar -C {DATA} && tar -xf {src}/images_nabirds.tar -C {DATA}")

aud = pd.read_csv(f"{DATA}/audio_ml.csv")
img = pd.read_csv(f"{DATA}/images_nabirds.csv")
aud["path"] = [f"{DATA}/audio_ml/{a}.wav" for a in aud.asset_id]
img["path"] = [f"{DATA}/images_nabirds/{a}.jpg" for a in img.asset_id]
aud = aud[[os.path.exists(p) for p in aud.path]].reset_index(drop=True)
img = img[[os.path.exists(p) for p in img.path]].reset_index(drop=True)
rng = np.random.default_rng(0)
sub = np.concatenate([rng.choice(g.index, min(IMG_PER_CLASS, len(g)), replace=False)
                      for _, g in img.groupby(["label", "split"])])
img_sub = img.loc[np.sort(sub)].reset_index(drop=True)
log(f"audio clips {len(aud)}  image subset {len(img_sub)} {img_sub.split.value_counts().to_dict()}")

# Quick check that each audio variant produces sane input, and is fast, before spending GPU time.
for v in ("base", "trim3", "slow2", "slow2_trim3", "win3"):
    t = time.time()
    segs = [audio_variant(p, v) for p in aud.path[:20]][0]
    per = (time.time() - t) / 20
    log(f"variant {v}: {len(segs)} segment(s), {len(segs[0]) / SR:.1f}s each, {1000 * per:.0f} ms/clip to prepare")
    assert per < 0.5, f"audio prep for {v} is too slow ({per:.2f}s/clip)"

jobs = [{"name": f"img_b{b}", "kind": "image", "budget": b, "items": list(img_sub.path)} for b in BUDGETS]
jobs += [{"name": f"aud_{v}", "kind": "audio", "variant": v, "items": list(aud.path)}
         for v in ("base", "trim3", "slow2", "slow2_trim3", "win3")]
json.dump(jobs, open(f"{WORK}/jobs.json", "w"))

nw = max(ngpu, 1)
t = time.time()
procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", str(i), str(nw)],
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i in range(nw)]
codes = [p.wait() for p in procs]
wall = time.time() - t
log(f"workers exited {codes} in {wall / 60:.1f} min")
assert all(c == 0 for c in codes), "a worker failed; see log above"

# Merge shards back into item order (window variants keep one row per window + owner index).
out, timing = {}, {"wall_min": round(wall / 60, 2), "gpus": nw, "per_job": {}}
for j in jobs:
    E = np.concatenate([np.load(f"{WORK}/{j['name']}.{s}.emb.npy") for s in range(nw)])
    O = np.concatenate([np.load(f"{WORK}/{j['name']}.{s}.own.npy") for s in range(nw)])
    order = np.argsort(O, kind="stable")
    out[j["name"]] = E[order].astype(np.float16)
    out[j["name"] + "__own"] = O[order]
for s in range(nw):
    for k, v in json.load(open(f"{WORK}/timing.{s}.json")).items():
        if k.startswith("_"):
            timing.setdefault("notes", []).extend(v)
            continue
        tj = timing["per_job"].setdefault(k, {"items": 0, "inputs": 0, "load_wait_s": 0, "encode_s": 0})
        for f in tj:
            tj[f] = round(tj[f] + v[f], 1)
for k, v in timing["per_job"].items():
    v["items_per_s_per_gpu"] = round(v["items"] / max(v["encode_s"] + v["load_wait_s"], 1e-9), 2)
    log(f"{k}: {v}")

np.savez_compressed(
    f"{OUT}/embed_v3.npz", **out,
    aud_id=aud.asset_id.astype(str).to_numpy(), aud_y=aud.label.to_numpy(), aud_split=aud.split.astype(str).to_numpy(),
    imgsub_id=img_sub.asset_id.astype(str).to_numpy(), imgsub_y=img_sub.label.to_numpy(),
    imgsub_split=img_sub.split.astype(str).to_numpy())
json.dump(timing, open(f"{OUT}/timing.json", "w"), indent=2)
log("done")
