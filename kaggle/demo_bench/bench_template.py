"""
Latency benchmark for the burds app (Kaggle, 2x T4 machine; nothing runs locally).

Runs the exact app code (embedded below by scripts/sync_kernels.sh) on 20 photos and 20 recordings,
once on a T4 GPU and once on CPU only (4 vCPUs), and times photo, song and photo + song requests
end to end: decode, preprocess, embed, classify, fuse. First call excluded (warm-up).

Outputs: latency.json, latency.md
"""
import base64
import glob
import json
import os
import subprocess
import sys
import tarfile
import time

OUT, WORK = "/kaggle/working", "/kaggle/temp/bench"
os.makedirs(WORK, exist_ok=True)
APP_B64 = "@@APP_B64@@"
N = 20

if os.environ.get("BURDS_BENCH_CHILD"):
    import numpy as np  # noqa: E402
    from PIL import Image  # noqa: E402
    sys.path.insert(0, WORK)
    import app  # noqa: E402  (loads the model on import)

    photos = sorted(glob.glob(f"{WORK}/images/**/*.jpg", recursive=True))[:N + 1]
    songs = sorted(glob.glob(f"{WORK}/audio/**/*.mp3", recursive=True))[:N + 1]
    res = {"device": app.DEVICE}

    def bench(name, fn, items):
        fn(items[0])  # warm-up
        t = []
        for it in items[1:]:
            t0 = time.perf_counter()
            fn(it)
            t.append(time.perf_counter() - t0)
        res[name] = {"n": len(t), "median_s": float(np.median(t)), "p90_s": float(np.percentile(t, 90))}
        print(app.DEVICE, name, res[name], flush=True)

    bench("photo", lambda p: app.identify(Image.open(p), None), photos)
    bench("song", lambda s: app.identify(None, s), songs)
    bench("photo + song", lambda ps: app.identify(Image.open(ps[0]), ps[1]), list(zip(photos, songs)))
    secs = [app.load_audio(s)[0].size / app.load_audio(s)[1] for s in songs[1:]]
    res["song_seconds_median"] = float(np.median(secs))
    json.dump(res, open(f"{WORK}/lat_{app.DEVICE}.json", "w"))
    sys.exit(0)


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", "set -euo pipefail; " + cmd], check=True)


sh("pip install -q -U 'gradio>=5' sentence-transformers transformers librosa")
open(f"{WORK}/app.py", "wb").write(base64.b64decode(APP_B64))
for t in glob.glob("/kaggle/input/**/images.tar", recursive=True) + glob.glob("/kaggle/input/**/audio_inat.tar", recursive=True):
    with tarfile.open(t) as tf:
        members = [m for m in tf.getmembers() if m.isfile()][: 4 * N]  # a few species' worth is plenty
        tf.extractall(WORK, members=members)

env = dict(os.environ, BURDS_BENCH_CHILD="1")
env["BURDS_DATA"] = os.path.dirname(glob.glob("/kaggle/input/**/demo_index.npz", recursive=True)[0])
env["BURDS_TAXA"] = [p for p in glob.glob("/kaggle/input/**/taxa.csv", recursive=True) if "demo-data" in p][0]
cfg = glob.glob("/kaggle/input/**/embeddinggemma*/**/config.json", recursive=True)
if cfg:
    env["BURDS_MODEL"] = os.path.dirname(sorted(cfg, key=len)[0])
script = os.path.abspath(__file__)
for dev in ("0", ""):  # T4, then CPU only
    subprocess.run([sys.executable, script], env=dict(env, CUDA_VISIBLE_DEVICES=dev), check=True)

R = {d: json.load(open(f"{WORK}/lat_{d}.json")) for d in ("cuda", "cpu")}
json.dump(R, open(f"{OUT}/latency.json", "w"), indent=2)
md = ["# burds latency (end to end per request, median / p90, first call excluded)", "",
      f"Song clips: median {R['cpu']['song_seconds_median']:.0f} s long. CPU = 4 vCPUs (Kaggle).", "",
      "| request | T4 GPU | CPU only |", "|---|---|---|"]
for k in ("photo", "song", "photo + song"):
    md.append(f"| {k} | {R['cuda'][k]['median_s']:.2f} s / {R['cuda'][k]['p90_s']:.2f} s | "
              f"{R['cpu'][k]['median_s']:.2f} s / {R['cpu'][k]['p90_s']:.2f} s |")
open(f"{OUT}/latency.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
