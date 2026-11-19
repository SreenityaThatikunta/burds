"""
More song training data: iNaturalist sound recordings (CPU + internet, no key needed).

xeno-canto recordings are mostly clean, focal recordings; iNaturalist sounds are mostly phone and
field recordings, closer to what a demo user (and SSW60's Macaulay clips) sound like. Up to
N_SOUNDS research-grade, CC0 / CC BY / CC BY-SA recordings per species, converted to mono mp3,
at most 60 s. Observations used in the demo video are excluded so it stays "never seen".

Outputs: audio_inat.tar, manifest_inat.csv, summary_inat.json
"""
import csv
import glob
import io
import json
import os
import subprocess
import tarfile
import time

import requests

T0 = time.time()
OUT, WORK = "/kaggle/working", "/kaggle/temp/snd"
os.makedirs(WORK, exist_ok=True)
N_SOUNDS, MAX_BYTES = 30, 25_000_000
LIC = ("cc0", "cc-by", "cc-by-sa")
S = requests.Session()
S.headers.update({"User-Agent": "burds-demo-data/0.1 (research demo)"})


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def get(params):
    for i in range(4):
        r = S.get("https://api.inaturalist.org/v1/observations", params=params, timeout=30)
        time.sleep(1.0)  # iNat asks for <= 1 req/s
        if r.ok:
            return r.json()["results"]
        time.sleep(5 * (i + 1))
    r.raise_for_status()


exclude = set()
for m in glob.glob("/kaggle/input/**/media.json", recursive=True):
    for sc in json.load(open(m)):
        exclude |= {sc[k]["url"] for k in ("photo_media", "song_media") if k in sc}
log(f"excluding {len(exclude)} demo-video observations")

taxa = list(csv.DictReader(open(glob.glob("/kaggle/input/**/burds-demo-data/taxa.csv", recursive=True)[0])))
fixes = {}
for f in glob.glob("/kaggle/input/**/taxon_fixes.json", recursive=True):
    fixes.update(json.load(open(f)))

rows = []
with tarfile.open(f"{OUT}/audio_inat.tar", "w") as tf:
    for t in taxa:
        code = t["species_code"]
        tid = fixes.get(code, {}).get("new_id") or t["inat_taxon_id"]
        obs = get({"taxon_id": tid, "sounds": "true", "quality_grade": "research", "sound_license": ",".join(LIC),
                   "per_page": 100, "order_by": "votes"})
        got = 0
        for o in obs:
            if got >= N_SOUNDS:
                break
            s = next((x for x in o.get("sounds", []) if x.get("license_code") in LIC and x.get("file_url")), None)
            if not s or o["uri"] in exclude:
                continue
            try:
                r = S.get(s["file_url"], timeout=120, stream=True)
                raw = r.raw.read(MAX_BYTES + 1, decode_content=True)
            except requests.RequestException:
                continue
            if not r.ok or len(raw) > MAX_BYTES:
                continue
            src, dst = f"{WORK}/raw", f"{WORK}/out.mp3"
            open(src, "wb").write(raw)
            if subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-t", "60", "-ac", "1", "-b:a", "128k",
                               dst]).returncode or os.path.getsize(dst) < 20000:
                continue
            fn = f"audio/{code}/inat_{s['id']}.mp3"
            data = open(dst, "rb").read()
            ti = tarfile.TarInfo(fn); ti.size = len(data)  # noqa: E702
            tf.addfile(ti, io.BytesIO(data))
            rows.append({"kind": "audio", "file": fn, "label": t["label"], "species_code": code,
                         "common_name": t["common_name"], "license": s["license_code"],
                         "author": s.get("attribution", ""), "source_url": o["uri"]})
            got += 1
        log(f"{code:8s} {t['common_name']:28s} inat sounds={got}")

with open(f"{OUT}/manifest_inat.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)  # noqa: E702
per = {}
for r in rows:
    per[r["species_code"]] = per.get(r["species_code"], 0) + 1
json.dump({"total": len(rows), "min": min(per.get(t["species_code"], 0) for t in taxa), "per_species": per,
           "seconds": round(time.time() - T0)}, open(f"{OUT}/summary_inat.json", "w"), indent=2)
log(f"done: {len(rows)} iNaturalist sounds")
