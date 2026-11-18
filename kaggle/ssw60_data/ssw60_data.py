"""
SSW60 data prep (CPU only, no GPU quota).

Streams the 31 GB SSW60 tarball from S3, keeps only the metadata CSVs, audio_ml/ and
images_nabirds/, and writes them to /kaggle/working as uncompressed tars + CSVs. Other
kernels attach this one as a `kernel_sources` input and skip the slow download.

Logs download progress (GB, %, MB/s, ETA) every 30 s so a slow transfer is distinguishable
from a stalled one. Output stays private on Kaggle: SSW60 media is research-only and must
not be redistributed.
"""
import os
import tarfile
import time
import urllib.request

URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/ssw60/ssw60.tar.gz"
OUT = "/kaggle/working"
KEEP = ("audio_ml/", "images_nabirds/")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


class Progress:
    """File-like wrapper that counts bytes read from the HTTP stream and logs periodically."""

    def __init__(self, f, total):
        self.f, self.total, self.n, self.last = f, total, 0, 0.0

    def read(self, size=-1):
        b = self.f.read(size)
        self.n += len(b)
        now = time.time()
        if now - self.last >= 30:
            self.last = now
            el = now - T0
            rate = self.n / el / 1e6
            pct = 100 * self.n / self.total if self.total else 0
            eta = (self.total - self.n) / (rate * 1e6) / 60 if rate and self.total else float("nan")
            log(f"downloaded {self.n / 1e9:6.2f} / {self.total / 1e9:.2f} GB ({pct:5.1f}%) "
                f"{rate:6.1f} MB/s  ETA {eta:5.1f} min")
        return b


req = urllib.request.Request(URL, headers={"User-Agent": "ssw60-prep"})
resp = urllib.request.urlopen(req, timeout=120)
total = int(resp.headers.get("Content-Length", 0))
log(f"streaming {URL} ({total / 1e9:.2f} GB)")

writers = {k.rstrip("/"): tarfile.open(f"{OUT}/{k.rstrip('/')}.tar", "w") for k in KEEP}
counts = {k: 0 for k in writers}
csvs = []
with tarfile.open(fileobj=Progress(resp, total), mode="r|gz") as src:
    for m in src:
        if not m.isfile():
            continue
        name = m.name
        base = os.path.basename(name)
        if base.endswith(".csv"):
            data = src.extractfile(m).read()
            with open(f"{OUT}/{base}", "wb") as f:
                f.write(data)
            csvs.append(name)
            log(f"csv: {name}")
            continue
        for k, w in writers.items():
            if f"/{k}/" in f"/{name}":
                m.name = f"{k}/{base}"  # flatten whatever top-level folder the archive uses
                w.addfile(m, src.extractfile(m))
                counts[k] += 1
                if counts[k] % 2000 == 0:
                    log(f"{k}: {counts[k]} files")
                break

for w in writers.values():
    w.close()
log(f"done: csvs={csvs} counts={counts}")
for f in sorted(os.listdir(OUT)):
    log(f"  {f}: {os.path.getsize(f'{OUT}/{f}') / 1e9:.2f} GB")
