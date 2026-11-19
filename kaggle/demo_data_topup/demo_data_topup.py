"""
Top up iNaturalist photos for species the first demo-data run came up short on (CPU + internet).

iNaturalist sometimes swaps taxa (e.g. Cooper's Hawk: Accipiter cooperii -> Astur cooperii), so
SSW60's taxon id can point at an inactive taxon with no new observations. For each species with
fewer than N_PHOTOS photos we resolve the current taxon and fetch the rest. No xeno-canto key
needed for photos. Songs for species xeno-canto files under a newer name (e.g. Hairy Woodpecker,
Leuconotopicus villosus) need the XENO_CANTO_KEY secret, so start that run from the Kaggle editor.

Outputs: images_topup.tar, manifest_topup.csv, taxon_fixes.json
"""
import csv
import glob
import io
import json
import tarfile
import time

import requests

T0 = time.time()
OUT = "/kaggle/working"
N_PHOTOS = 30
S = requests.Session()
S.headers.update({"User-Agent": "burds-demo-data/0.1 (research demo)"})
LIC = ("cc0", "cc-by", "cc-by-sa")


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def get(url, params=None):
    for i in range(4):
        r = S.get(url, params=params, timeout=30)
        if r.ok:
            time.sleep(1.0)  # iNat asks for <= 1 req/s
            return r.json()
        time.sleep(5 * (i + 1))
    r.raise_for_status()


def current_taxon(tid, sci):
    t = get(f"https://api.inaturalist.org/v1/taxa/{tid}")["results"][0]
    if t.get("is_active"):
        return tid, t["name"]
    syn = t.get("current_synonymous_taxon_ids") or []
    if syn:
        n = get(f"https://api.inaturalist.org/v1/taxa/{syn[0]}")["results"][0]
        return syn[0], n["name"]
    hits = [h for h in get("https://api.inaturalist.org/v1/taxa", {"q": sci, "rank": "species"})["results"]
            if h.get("is_active")]
    return (hits[0]["id"], hits[0]["name"]) if hits else (tid, sci)


taxa = {t["species_code"]: t for t in csv.DictReader(open(glob.glob("/kaggle/input/**/taxa.csv", recursive=True)[0]))}
man = list(csv.DictReader(open(glob.glob("/kaggle/input/**/manifest.csv", recursive=True)[0])))
have = {c: 0 for c in taxa}
for r in man:
    if r["kind"] == "image":
        have[r["species_code"]] += 1
short = {c: n for c, n in have.items() if n < N_PHOTOS}
log(f"species short on photos: {short}")

rows, fixes = [], {}
with tarfile.open(f"{OUT}/images_topup.tar", "w") as tf:
    for code, n in short.items():
        t = taxa[code]
        tid, name = current_taxon(int(t["inat_taxon_id"]), t["scientific_name"])
        fixes[code] = {"old_id": int(t["inat_taxon_id"]), "new_id": tid, "name": name}
        d = get("https://api.inaturalist.org/v1/observations", {
            "taxon_id": tid, "quality_grade": "research", "photos": "true",
            "photo_license": ",".join(LIC), "per_page": 100, "order_by": "votes"})
        got = 0
        for o in d["results"]:
            if n + got >= N_PHOTOS:
                break
            ph = next((p for p in o.get("photos", []) if p.get("license_code") in LIC), None)
            if not ph:
                continue
            r = S.get(ph["url"].replace("/square.", "/medium."), timeout=60)
            if not r.ok:
                continue
            fn = f"images/{code}/inat_{ph['id']}.jpg"
            ti = tarfile.TarInfo(fn); ti.size = len(r.content)  # noqa: E702
            tf.addfile(ti, io.BytesIO(r.content))
            rows.append({"kind": "image", "file": fn, "label": t["label"], "species_code": code,
                         "common_name": t["common_name"], "license": ph["license_code"],
                         "author": ph.get("attribution", ""), "source_url": o["uri"]})
            got += 1
        log(f"{code}: taxon {fixes[code]['old_id']} -> {tid} ({name}), +{got} photos")

# ---------------------------------------------------------------- songs for species xeno-canto lists under
# another name (needs the XENO_CANTO_KEY secret, so this part only runs when started from the editor)
N_SONGS, MAX_SECS = 20, 90
try:
    from kaggle_secrets import UserSecretsClient
    XC_KEY = UserSecretsClient().get_secret("XENO_CANTO_KEY")
except Exception as e:  # noqa: BLE001
    XC_KEY = None
    log(f"no xeno-canto key ({e}); skipping song top-up")

songs = {c: 0 for c in taxa}
for r in man:
    if r["kind"] == "audio":
        songs[r["species_code"]] += 1
short_a = {c: n for c, n in songs.items() if n < 15}
log(f"species short on songs: {short_a}")


def secs(rec):
    try:
        p = [int(x) for x in rec.get("length", "99:00").split(":")]
        return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]
    except ValueError:
        return 9999


n_aud = 0
if XC_KEY and short_a:
    with tarfile.open(f"{OUT}/audio_topup.tar", "w") as tf:
        for code, n in short_a.items():
            t = taxa[code]
            names = [t["scientific_name"]]
            if code in fixes:
                names.append(fixes[code]["name"])
            else:
                names.append(current_taxon(int(t["inat_taxon_id"]), t["scientific_name"])[1])
            cand, used = [], None
            for nm in dict.fromkeys(names):
                gen, sp = nm.lower().split()[:2]
                for q in ("A", "B"):
                    r = S.get("https://xeno-canto.org/api/3/recordings", timeout=30,
                              params={"query": f"gen:{gen} sp:{sp} q:{q}", "key": XC_KEY, "per_page": 100})
                    time.sleep(1.0)
                    cand += r.json().get("recordings", []) if r.ok else []
                if cand:
                    used = nm
                    break
            cand = [c for c in cand if c.get("file", "").startswith("http") and 3 <= secs(c) <= MAX_SECS]
            cand.sort(key=lambda c: (c.get("q", "Z"), "song" not in c.get("type", "")))
            got = 0
            for c in cand[:N_SONGS - n]:
                r = S.get(c["file"], timeout=120)
                if not r.ok:
                    continue
                fn = f"audio/{code}/xc_{c['id']}.mp3"
                ti = tarfile.TarInfo(fn); ti.size = len(r.content)  # noqa: E702
                tf.addfile(ti, io.BytesIO(r.content))
                rows.append({"kind": "audio", "file": fn, "label": t["label"], "species_code": code,
                             "common_name": t["common_name"], "license": c.get("lic", ""),
                             "author": c.get("rec", ""), "source_url": c.get("url", "")})
                got += 1
            n_aud += got
            fixes.setdefault(code, {})["xc_name"] = used
            log(f"{code}: xeno-canto name {used}, +{got} songs")

with open(f"{OUT}/manifest_topup.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(man[0]))
    w.writeheader(); w.writerows(rows)  # noqa: E702
json.dump(fixes, open(f"{OUT}/taxon_fixes.json", "w"), indent=2)
log(f"done: +{len(rows) - n_aud} photos for {len(short)} species, +{n_aud} songs for {len(short_a)} species")
