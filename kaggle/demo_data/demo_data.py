"""
Collect a redistributable reference set for the public demo (CPU + internet, runs on Kaggle).

SSW60 is research-only, so the demo's "memory" is built from openly licensed media for the same
60 species:
  photos  - iNaturalist research-grade observations, photo licence CC0 / CC BY / CC BY-SA
  songs   - xeno-canto recordings (all Creative Commons; some are NC, fine for a non-commercial demo)
Every file is logged with its licence and author so the app can credit it.

Needs the Kaggle secret XENO_CANTO_KEY (https://xeno-canto.org/account).
Outputs (/kaggle/working): images.tar, audio.tar, manifest.csv, taxa.csv, summary.json
"""
import csv
import glob
import io
import json
import os
import tarfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import requests

T0 = time.time()
OUT = "/kaggle/working"
N_PHOTOS, N_SONGS, MAX_SECS = 30, 20, 90
UA = {"User-Agent": "burds-demo-data/0.1 (research demo)"}
S = requests.Session()
S.headers.update(UA)


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def get_json(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(10 * (i + 1)); continue  # noqa: E702
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log(f"  retry {i + 1} {url}: {e}")
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"failed: {url} {params}")


def fetch(url, tries=3):
    for i in range(tries):
        try:
            r = S.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    return None


try:
    from kaggle_secrets import UserSecretsClient
    XC_KEY = UserSecretsClient().get_secret("XENO_CANTO_KEY")
except Exception as e:  # noqa: BLE001
    XC_KEY = os.environ.get("XENO_CANTO_KEY")
    if not XC_KEY:
        raise SystemExit(f"XENO_CANTO_KEY secret not available to this kernel ({e}). "
                         "Attach it in the notebook editor: Add-ons -> Secrets.")
r = S.get("https://xeno-canto.org/api/3/recordings",
          params={"query": 'sp:"turdus migratorius"', "key": XC_KEY, "per_page": 50}, timeout=30)
assert r.ok and "recordings" in r.json(), f"xeno-canto key check failed: {r.status_code} {r.text[:200]}"
log("xeno-canto key ok")

taxa = list(csv.DictReader(open(glob.glob("/kaggle/input/**/taxa.csv", recursive=True)[0])))
assert len(taxa) == 60, len(taxa)

rows = []  # manifest
img_jobs, aud_jobs = [], []
for t in taxa:
    code, sci = t["species_code"], t["scientific_name"]
    # --- photos: one photo per observation, most-faved first for clear, well-framed shots
    seen, obs_page = 0, 1
    while seen < N_PHOTOS and obs_page <= 3:
        d = get_json("https://api.inaturalist.org/v1/observations", {
            "taxon_id": t["inat_taxon_id"], "quality_grade": "research", "photos": "true",
            "photo_license": "cc0,cc-by,cc-by-sa", "per_page": 100, "page": obs_page, "order_by": "votes"})
        for o in d["results"]:
            ph = next((p for p in o.get("photos", []) if p.get("license_code") in ("cc0", "cc-by", "cc-by-sa")), None)
            if not ph or seen >= N_PHOTOS:
                continue
            fn = f"images/{code}/inat_{ph['id']}.jpg"
            img_jobs.append((ph["url"].replace("/square.", "/medium."), fn))
            rows.append({"kind": "image", "file": fn, "label": t["label"], "species_code": code,
                         "common_name": t["common_name"], "license": ph["license_code"],
                         "author": ph.get("attribution", ""), "source_url": o["uri"]})
            seen += 1
        obs_page += 1
        time.sleep(1.0)  # iNat asks for <= 1 req/s
    # --- songs: quality A first, then B; songs before calls; skip long recordings
    gen, sp = sci.split()[0].lower(), sci.split()[1].lower()
    cand = []
    for q in ("A", "B"):
        d = get_json("https://xeno-canto.org/api/3/recordings",
                     {"query": f'gen:{gen} sp:{sp} q:{q}', "key": XC_KEY, "per_page": 100})
        cand += d.get("recordings", [])
        if len(cand) >= 3 * N_SONGS:
            break
        time.sleep(1.0)

    def secs(rec):
        try:
            p = [int(x) for x in rec.get("length", "99:00").split(":")]
            return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]
        except ValueError:
            return 9999

    cand = [c for c in cand if c.get("file", "").startswith("http") and 3 <= secs(c) <= MAX_SECS]
    cand.sort(key=lambda c: (c.get("q", "Z"), "song" not in c.get("type", "")))
    for c in cand[:N_SONGS]:
        fn = f"audio/{code}/xc_{c['id']}.mp3"
        aud_jobs.append((c["file"], fn))
        rows.append({"kind": "audio", "file": fn, "label": t["label"], "species_code": code,
                     "common_name": t["common_name"], "license": c.get("lic", ""),
                     "author": c.get("rec", ""), "source_url": c.get("url", "")})
    log(f"{code:8s} {t['common_name']:28s} photos={seen:2d} songs={min(len(cand), N_SONGS):2d}")


def download_into(tar_path, jobs, workers):
    ok = set()
    with tarfile.open(tar_path, "w") as tf, ThreadPoolExecutor(workers) as ex:
        for i, ((url, fn), data) in enumerate(zip(jobs, ex.map(lambda j: fetch(j[0]), jobs))):
            if data:
                ti = tarfile.TarInfo(fn); ti.size = len(data)  # noqa: E702
                tf.addfile(ti, io.BytesIO(data)); ok.add(fn)  # noqa: E702
            if i % 200 == 0:
                log(f"  {tar_path.split('/')[-1]}: {i}/{len(jobs)}")
    return ok


got = download_into(f"{OUT}/images.tar", img_jobs, 16)
got |= download_into(f"{OUT}/audio.tar", aud_jobs, 4)  # be gentle with xeno-canto
rows = [r for r in rows if r["file"] in got]
with open(f"{OUT}/manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)  # noqa: E702
with open(f"{OUT}/taxa.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(taxa[0]))
    w.writeheader(); w.writerows(taxa)  # noqa: E702
per = {}
for r in rows:
    per.setdefault(r["species_code"], {"image": 0, "audio": 0})[r["kind"]] += 1
summary = {"images": sum(v["image"] for v in per.values()), "audio": sum(v["audio"] for v in per.values()),
           "min_images": min(v["image"] for v in per.values()), "min_audio": min(v["audio"] for v in per.values()),
           "per_species": per, "seconds": round(time.time() - T0)}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
log(f"done: {summary['images']} photos, {summary['audio']} songs "
    f"(min per species {summary['min_images']}/{summary['min_audio']})")
