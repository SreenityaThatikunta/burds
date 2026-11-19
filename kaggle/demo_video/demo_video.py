"""
Record a demo video of the burds app (GPU + internet, runs on Kaggle; nothing runs locally).

1. Pick openly licensed media the model has never seen: iNaturalist photos and *sound* recordings
   (training used iNaturalist photos + xeno-canto songs; every training observation is excluded).
2. Start the exact app from app/app.py (embedded below by scripts/sync_kernels.sh) on the GPU.
3. Drive it with headless Chromium (Playwright): upload, click Identify, open the breakdown;
   record the screen, then convert to MP4 + GIF with ffmpeg.

Outputs: burds_demo.mp4, burds_demo.gif, scene*.png, media.json (credits + predictions)
"""
import base64
import csv
import glob
import json
import os
import subprocess
import sys
import time

import requests

T0 = time.time()
OUT, WORK = "/kaggle/working", "/kaggle/temp/video"
os.makedirs(WORK, exist_ok=True)
APP_B64 = "IiIiCmJ1cmRzOiBpZGVudGlmeSBhIE5vcnRoIEFtZXJpY2FuIGJpcmQgZnJvbSBhIHBob3RvLCBhIHNvbmcsIG9yIGJvdGguCgpObyB0cmFpbmluZyBvbiB0aGUgbW9kZWwgaXRzZWxmOiBFbWJlZGRpbmdHZW1tYSAyIHR1cm5zIHRoZSBwaG90byBhbmQgdGhlIHNvbmcgaW50byB2ZWN0b3JzLCBhbmQKdHdvIHNtYWxsIGxpbmVhciBjbGFzc2lmaWVycyAoNjAgc3BlY2llcywgdHJhaW5lZCBvbiBvcGVubHkgbGljZW5zZWQgaU5hdHVyYWxpc3QgcGhvdG9zIGFuZAp4ZW5vLWNhbnRvIHJlY29yZGluZ3MpIHJlYWQgdGhvc2UgdmVjdG9ycy4gUGhvdG8gYW5kIHNvbmcgcHJlZGljdGlvbnMgYXJlIGNvbWJpbmVkIGJ5IGNvbmZpZGVuY2UuCiIiIgppbXBvcnQgY3N2CmltcG9ydCBvcwppbXBvcnQgdGltZQoKaW1wb3J0IGdyYWRpbyBhcyBncgppbXBvcnQgbnVtcHkgYXMgbnAKaW1wb3J0IHRvcmNoCmZyb20gUElMIGltcG9ydCBJbWFnZQpmcm9tIHNlbnRlbmNlX3RyYW5zZm9ybWVycyBpbXBvcnQgU2VudGVuY2VUcmFuc2Zvcm1lcgoKSEVSRSA9IG9zLmVudmlyb24uZ2V0KCJCVVJEU19EQVRBIikgb3Igb3MucGF0aC5kaXJuYW1lKG9zLnBhdGguYWJzcGF0aChfX2ZpbGVfXykpClNSID0gMTYwMDAKSURYID0gbnAubG9hZChvcy5wYXRoLmpvaW4oSEVSRSwgImRlbW9faW5kZXgubnB6IikpCkNPTU1PTiwgU0NJID0gSURYWyJjb21tb24iXS50b2xpc3QoKSwgSURYWyJzY2kiXS50b2xpc3QoKQpDT0RFUyA9IFtyWyJzcGVjaWVzX2NvZGUiXSBmb3IgciBpbiBjc3YuRGljdFJlYWRlcihvcGVuKG9zLmVudmlyb24uZ2V0KCJCVVJEU19UQVhBIikgb3Igb3MucGF0aC5qb2luKEhFUkUsICJ0YXhhLmNzdiIpKSldCldfSU1HLCBCX0lNRywgV19BVUQsIEJfQVVEID0gSURYWyJXX2ltZyJdLCBJRFhbImJfaW1nIl0sIElEWFsiV19hdWQiXSwgSURYWyJiX2F1ZCJdCkdBTU1BLCBCVURHRVQgPSBmbG9hdChJRFhbImdhbW1hIl0pLCBpbnQoSURYWyJidWRnZXQiXSkKQVVESU9fS0lORFMgPSBJRFhbImF1ZGlvX2tpbmRzIl0udG9saXN0KCkKCnRvcmNoLnNldF9udW1fdGhyZWFkcyhvcy5jcHVfY291bnQoKSBvciAyKQpERVZJQ0UgPSAiY3VkYSIgaWYgdG9yY2guY3VkYS5pc19hdmFpbGFibGUoKSBlbHNlICJjcHUiCk1PREVMID0gU2VudGVuY2VUcmFuc2Zvcm1lcihvcy5lbnZpcm9uLmdldCgiQlVSRFNfTU9ERUwiLCAiZ29vZ2xlL2VtYmVkZGluZ2dlbW1hLTIiKSwgZGV2aWNlPURFVklDRSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1vZGVsX2t3YXJncz17InRvcmNoX2R0eXBlIjogdG9yY2guZmxvYXQzMn0pCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIGF1ZGlvIChpZGVudGljYWwgdG8gdGhlIEthZ2dsZSByZWNpcGUpCmRlZiByZXNhbXBsZSh4LCBzcl9pbiwgc3Jfb3V0KToKICAgIGZyb20gbWF0aCBpbXBvcnQgZ2NkCiAgICBmcm9tIHNjaXB5LnNpZ25hbCBpbXBvcnQgcmVzYW1wbGVfcG9seQogICAgc3JfaW4sIHNyX291dCA9IGludChyb3VuZChzcl9pbikpLCBpbnQocm91bmQoc3Jfb3V0KSkKICAgIGcgPSBnY2Qoc3JfaW4sIHNyX291dCkKICAgIHJldHVybiByZXNhbXBsZV9wb2x5KHgsIHNyX291dCAvLyBnLCBzcl9pbiAvLyBnKS5hc3R5cGUobnAuZmxvYXQzMikKCgpkZWYgbG91ZGVzdCh4LCBzciwgc2Vjcyk6CiAgICBmcm9tIHNjaXB5LnNpZ25hbCBpbXBvcnQgYnV0dGVyLCBzb3NmaWx0CiAgICBuID0gaW50KHNlY3MgKiBzcikKICAgIGlmIGxlbih4KSA8PSBuOgogICAgICAgIHJldHVybiAwCiAgICBocCA9IHNvc2ZpbHQoYnV0dGVyKDQsIDEwMDAsICJoaWdocGFzcyIsIGZzPXNyLCBvdXRwdXQ9InNvcyIpLCB4KQogICAgYyA9IG5wLmNvbmNhdGVuYXRlKFtbMC4wXSwgbnAuY3Vtc3VtKGhwLmFzdHlwZShucC5mbG9hdDY0KSAqKiAyKV0pCiAgICByZXR1cm4gaW50KG5wLmFyZ21heChjW246XSAtIGNbOi1uXSkpCgoKZGVmIGF1ZGlvX3ZhcmlhbnQoeCwgc3IsIGtpbmQpOgogICAgIiIiSGFsZiBzcGVlZCByZWFkcyB0aGUgc2FtcGxlcyBhdCBzci8yLCBzbyBhIDEwIGtIeiBub3RlIGxhbmRzIGF0IDUga0h6LCB3aGljaCB0aGUgbW9kZWwgY2FuIGhlYXIuIiIiCiAgICByYXRlID0gc3IgLyAyIGlmIGtpbmQuc3RhcnRzd2l0aCgic2xvdzIiKSBlbHNlIHNyCiAgICBuID0gZmxvYXQoa2luZC5zcGxpdCgidHJpbSIpWzFdKQogICAgcyA9IGxvdWRlc3QoeCwgc3IsIG4pCiAgICByZXR1cm4gcmVzYW1wbGUoeFtzOnMgKyBpbnQobiAqIHNyKV0sIHJhdGUsIFNSKQoKCmRlZiBsb2FkX2F1ZGlvKHBhdGgpOgogICAgaW1wb3J0IGxpYnJvc2EKICAgIHgsIHNyID0gbGlicm9zYS5sb2FkKHBhdGgsIHNyPU5vbmUsIG1vbm89VHJ1ZSwgZHVyYXRpb249MTIwKQogICAgcmV0dXJuIG5wLmFzYXJyYXkoeCwgbnAuZmxvYXQzMiksIHNyCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIHByZWRpY3Rpb24KZGVmIHNvZnRtYXgoeik6CiAgICB6ID0geiAtIHoubWF4KCkKICAgIGUgPSBucC5leHAoeikKICAgIHJldHVybiBlIC8gZS5zdW0oKQoKCmRlZiBibG9ja3Nfbm9ybSh2LCBkPTc2OCk6CiAgICB2ID0gdi5yZXNoYXBlKC0xLCBkKQogICAgcmV0dXJuICh2IC8gbnAubGluYWxnLm5vcm0odiwgYXhpcz0xLCBrZWVwZGltcz1UcnVlKS5jbGlwKDFlLTEyKSkucmVzaGFwZSgtMSkKCgpkZWYgcGhvdG9fcHJvYnMoaW1nKToKICAgIGUgPSBNT0RFTC5lbmNvZGUoW3siaW1hZ2UiOiBpbWcuY29udmVydCgiUkdCIil9XSwgbm9ybWFsaXplX2VtYmVkZGluZ3M9VHJ1ZSwgY29udmVydF90b19udW1weT1UcnVlLAogICAgICAgICAgICAgICAgICAgICBwcm9jZXNzaW5nX2t3YXJncz17ImltYWdlIjogeyJtYXhfc29mdF90b2tlbnMiOiBCVURHRVR9fSlbMF0KICAgIHJldHVybiBzb2Z0bWF4KFdfSU1HIEAgZSArIEJfSU1HKQoKCmRlZiBzb25nX3Byb2JzKHBhdGgpOgogICAgeCwgc3IgPSBsb2FkX2F1ZGlvKHBhdGgpCiAgICBpZiBsZW4oeCkgPCBzcjoKICAgICAgICByYWlzZSBnci5FcnJvcigiVGhlIHJlY29yZGluZyBpcyBzaG9ydGVyIHRoYW4gMSBzZWNvbmQuIikKICAgIGUgPSBNT0RFTC5lbmNvZGUoW3siYXVkaW8iOiBhdWRpb192YXJpYW50KHgsIHNyLCBrKX0gZm9yIGsgaW4gQVVESU9fS0lORFNdLAogICAgICAgICAgICAgICAgICAgICBub3JtYWxpemVfZW1iZWRkaW5ncz1UcnVlLCBjb252ZXJ0X3RvX251bXB5PVRydWUpCiAgICByZXR1cm4gc29mdG1heChXX0FVRCBAIGJsb2Nrc19ub3JtKGUucmVzaGFwZSgtMSkpICsgQl9BVUQpCgoKZGVmIGZ1c2UocGksIHBhKToKICAgIGEsIGIgPSBwaS5tYXgoKSAqKiBHQU1NQSwgcGEubWF4KCkgKiogR0FNTUEKICAgIHJldHVybiBzb2Z0bWF4KChhICogbnAubG9nKHBpICsgMWUtOSkgKyBiICogbnAubG9nKHBhICsgMWUtOSkpIC8gKGEgKyBiKSkKCgpkZWYgdG9wKHAsIGs9NSk6CiAgICByZXR1cm4ge2Yie0NPTU1PTltpXX0gKHtTQ0lbaV19KSI6IGZsb2F0KHBbaV0pIGZvciBpIGluIG5wLmFyZ3NvcnQoLXApWzprXX0KCgpkZWYgaWRlbnRpZnkoaW1nLCBhdWRpbyk6CiAgICBpZiBpbWcgaXMgTm9uZSBhbmQgYXVkaW8gaXMgTm9uZToKICAgICAgICByYWlzZSBnci5FcnJvcigiQWRkIGEgcGhvdG8sIGEgcmVjb3JkaW5nLCBvciBib3RoLiIpCiAgICBwaSA9IHBob3RvX3Byb2JzKGltZykgaWYgaW1nIGlzIG5vdCBOb25lIGVsc2UgTm9uZQogICAgcGEgPSBzb25nX3Byb2JzKGF1ZGlvKSBpZiBhdWRpbyBpcyBub3QgTm9uZSBlbHNlIE5vbmUKICAgIHAgPSBmdXNlKHBpLCBwYSkgaWYgcGkgaXMgbm90IE5vbmUgYW5kIHBhIGlzIG5vdCBOb25lIGVsc2UgKHBpIGlmIHBpIGlzIG5vdCBOb25lIGVsc2UgcGEpCiAgICBiZXN0ID0gaW50KHAuYXJnbWF4KCkpCiAgICB0aHVtYiA9IG9zLnBhdGguam9pbihIRVJFLCAidGh1bWJzIiwgZiJ7Q09ERVNbYmVzdF19LmpwZyIpCiAgICByZXR1cm4gKHRvcChwKSwgdG9wKHBpKSBpZiBwaSBpcyBub3QgTm9uZSBlbHNlIE5vbmUsIHRvcChwYSkgaWYgcGEgaXMgbm90IE5vbmUgZWxzZSBOb25lLAogICAgICAgICAgICBJbWFnZS5vcGVuKHRodW1iKS5jb252ZXJ0KCJSR0IiKSBpZiBvcy5wYXRoLmV4aXN0cyh0aHVtYikgZWxzZSBOb25lKQoKCkVYID0gb3MucGF0aC5qb2luKEhFUkUsICJleGFtcGxlcyIpCmV4YW1wbGVzID0gW1tvcy5wYXRoLmpvaW4oRVgsIGEpIGlmIGEgZWxzZSBOb25lLCBvcy5wYXRoLmpvaW4oRVgsIGIpIGlmIGIgZWxzZSBOb25lXQogICAgICAgICAgICBmb3IgYSwgYiBpbiAoKCJwaG90bzEuanBnIiwgTm9uZSksIChOb25lLCAic29uZzEubXAzIiksICgicGhvdG8xLmpwZyIsICJzb25nMS5tcDMiKSkKICAgICAgICAgICAgaWYgYWxsKG9zLnBhdGguZXhpc3RzKG9zLnBhdGguam9pbihFWCwgZikpIGZvciBmIGluIChhLCBiKSBpZiBmKV0KCndpdGggZ3IuQmxvY2tzKHRpdGxlPSJidXJkczogYmlyZCBJRCBmcm9tIHBob3RvICsgc29uZyIpIGFzIGRlbW86CiAgICBnci5NYXJrZG93bigKICAgICAgICAiIyBidXJkc1xuSWRlbnRpZnkgb25lIG9mICoqNjAgTm9ydGggQW1lcmljYW4gYmlyZHMqKiBmcm9tIGEgcGhvdG8sIGEgc29uZywgb3IgYm90aC4gIgogICAgICAgICJPbmUgbW9kZWwgKFtFbWJlZGRpbmdHZW1tYSAyXShodHRwczovL2h1Z2dpbmdmYWNlLmNvL2dvb2dsZS9lbWJlZGRpbmdnZW1tYS0yKSkgZW1iZWRzIGJvdGgsICIKICAgICAgICAid2l0aCBubyBmaW5lLXR1bmluZzogc21hbGwgbGluZWFyIGNsYXNzaWZpZXJzIHRyYWluZWQgb24gb3Blbmx5IGxpY2Vuc2VkIGlOYXR1cmFsaXN0IHBob3RvcyBhbmQgIgogICAgICAgICJ4ZW5vLWNhbnRvIHJlY29yZGluZ3MgZG8gdGhlIHJlc3QuIFtIb3cgaXQgd29ya3MgYW5kIHJlc3VsdHNdKGh0dHBzOi8vZ2l0aHViLmNvbS9TcmVlbml0eWFUaGF0aWt1bnRhL2J1cmRzKSIpCiAgICB3aXRoIGdyLlJvdygpOgogICAgICAgIHdpdGggZ3IuQ29sdW1uKCk6CiAgICAgICAgICAgIGltZyA9IGdyLkltYWdlKHR5cGU9InBpbCIsIGxhYmVsPSJQaG90byIsIHNvdXJjZXM9WyJ1cGxvYWQiLCAid2ViY2FtIiwgImNsaXBib2FyZCJdLCBlbGVtX2lkPSJwaG90byIsIGhlaWdodD0zMDApCiAgICAgICAgICAgIGF1ZCA9IGdyLkF1ZGlvKHR5cGU9ImZpbGVwYXRoIiwgbGFiZWw9IlNvbmcgb3IgY2FsbCIsIHNvdXJjZXM9WyJ1cGxvYWQiLCAibWljcm9waG9uZSJdLCBlbGVtX2lkPSJzb25nIikKICAgICAgICAgICAgYnRuID0gZ3IuQnV0dG9uKCJJZGVudGlmeSIsIHZhcmlhbnQ9InByaW1hcnkiLCBlbGVtX2lkPSJpZGVudGlmeSIpCiAgICAgICAgd2l0aCBnci5Db2x1bW4oKToKICAgICAgICAgICAgb3V0ID0gZ3IuTGFiZWwobnVtX3RvcF9jbGFzc2VzPTUsIGxhYmVsPSJCZXN0IGd1ZXNzIiwgZWxlbV9pZD0iYmVzdCIpCiAgICAgICAgICAgIHJlZiA9IGdyLkltYWdlKGxhYmVsPSJSZWZlcmVuY2UgcGhvdG8gKGlOYXR1cmFsaXN0LCBzZWUgY3JlZGl0cykiLCBpbnRlcmFjdGl2ZT1GYWxzZSwgaGVpZ2h0PTIyMCkKICAgICAgICAgICAgd2l0aCBnci5BY2NvcmRpb24oIlBob3RvIG9ubHkgLyBzb25nIG9ubHkiLCBvcGVuPUZhbHNlLCBlbGVtX2lkPSJzcGxpdCIpOgogICAgICAgICAgICAgICAgb3V0X2kgPSBnci5MYWJlbChudW1fdG9wX2NsYXNzZXM9NSwgbGFiZWw9IkZyb20gdGhlIHBob3RvIikKICAgICAgICAgICAgICAgIG91dF9hID0gZ3IuTGFiZWwobnVtX3RvcF9jbGFzc2VzPTUsIGxhYmVsPSJGcm9tIHRoZSBzb25nIikKICAgIGlmIGV4YW1wbGVzOgogICAgICAgIGdyLkV4YW1wbGVzKGV4YW1wbGVzLCBbaW1nLCBhdWRdKQogICAgZ3IuTWFya2Rvd24oZiJSdW5uaW5nIG9uIHtERVZJQ0UudXBwZXIoKX07IGEgcHJlZGljdGlvbiB0YWtlcyBhIGZldyBzZWNvbmRzIG9uIENQVSwgdW5kZXIgb25lIG9uIEdQVS4gU3BlY2llcyBsaXN0IGFuZCBtZWRpYSBjcmVkaXRzOiAiCiAgICAgICAgICAgICAgICAiYHRheGEuY3N2YCwgYGNyZWRpdHMuY3N2YCBpbiB0aGlzIFNwYWNlLiIpCiAgICBidG4uY2xpY2soaWRlbnRpZnksIFtpbWcsIGF1ZF0sIFtvdXQsIG91dF9pLCBvdXRfYSwgcmVmXSkKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBpZiBvcy5lbnZpcm9uLmdldCgiQlVSRFNfU0hBUkUiKTogICMgdGVtcG9yYXJ5IHB1YmxpYyBsaW5rIChlLmcuIGZyb20gYSBLYWdnbGUgR1BVIHJ1bikKICAgICAgICBkZW1vLmxhdW5jaChzaGFyZT1UcnVlLCBwcmV2ZW50X3RocmVhZF9sb2NrPVRydWUpCiAgICAgICAgcHJpbnQoZiJCVVJEU19MSVZFIHNoYXJlIGxpbmsgYWJvdmU7IHN0YXlpbmcgdXAgZm9yIHtvcy5lbnZpcm9uLmdldCgnQlVSRFNfSE9VUlMnLCAnMycpfSBoIiwgZmx1c2g9VHJ1ZSkKICAgICAgICB0aW1lLnNsZWVwKGZsb2F0KG9zLmVudmlyb24uZ2V0KCJCVVJEU19IT1VSUyIsICIzIikpICogMzYwMCkKICAgIGVsc2U6CiAgICAgICAgZGVtby5sYXVuY2goKQo="
LIC = "cc0,cc-by,cc-by-sa"
S = requests.Session()
S.headers.update({"User-Agent": "burds-demo-video/0.1 (research demo)"})


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def sh(cmd):
    log(f"$ {cmd}")
    subprocess.run(["bash", "-c", "set -euo pipefail; " + cmd], check=True)


sh("pip install -q -U 'gradio>=5' sentence-transformers transformers librosa playwright")
sh("python -m playwright install --with-deps chromium > /dev/null")

# ---------------------------------------------------------------- unseen media from iNaturalist
used = set()
for m in glob.glob("/kaggle/input/**/credits.csv", recursive=True):
    used |= {r["source_url"] for r in csv.DictReader(open(m))}
log(f"{len(used)} training sources excluded")


def inat(params):
    for i in range(4):
        r = S.get("https://api.inaturalist.org/v1/observations", params=params, timeout=30)
        time.sleep(1.0)
        if r.ok:
            return r.json()["results"]
        time.sleep(5 * (i + 1))
    r.raise_for_status()


def candidates(sci, common, kind, n=8):
    """Up to n unseen, openly licensed iNaturalist observations with a photo or a sound."""
    q = {"photos": "true", "photo_license": LIC} if kind == "photo" else {"sounds": "true", "sound_license": LIC}
    obs = []
    for name in (sci, common):
        obs = inat({"taxon_name": name, "iconic_taxa": "Aves", "quality_grade": "research", "per_page": 100, "order_by": "votes",
                    "page": 2 if kind == "photo" else 1, **q})
        if obs:
            break
    out = []
    for o in obs:
        media = o["photos"] if kind == "photo" else o["sounds"]
        m = next((x for x in media if x.get("license_code") in LIC.split(",")), None)
        if m and o["uri"] not in used and (kind == "photo" or m.get("file_url")):
            out.append((o, m))
        if len(out) == n:
            break
    return out


def fetch_media(o, m, kind):
    if kind == "photo":
        path = f"{WORK}/photo_{m['id']}.jpg"
        open(path, "wb").write(S.get(m["url"].replace("/square.", "/large."), timeout=60).content)
    else:
        raw, path = f"{WORK}/raw_{m['id']}", f"{WORK}/song_{m['id']}.mp3"
        open(raw, "wb").write(S.get(m["file_url"], timeout=120).content)
        # mp3, mono, at most 30 s: plays in any browser and keeps the video short
        if subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-t", "30", "-ac", "1", "-b:a", "128k",
                           path]).returncode or os.path.getsize(path) < 20000:
            return None
    who = m.get("attribution", "")
    who = who if who.startswith("(c)") else m["license_code"].upper()
    return {"file": path, "author": who, "license": m["license_code"], "url": o["uri"]}


# Scenes: (scientific name, common name) candidates in order of preference. Each scene uses the
# first unseen example the app gets right; how many were tried is saved in media.json.
SCENES = [
    {"title": "1 · Photo only", "kinds": ["photo"],
     "species": [("Cardinalis cardinalis", "Northern Cardinal"), ("Cyanocitta cristata", "Blue Jay")]},
    {"title": "2 · Song only", "kinds": ["song"],
     "species": [("Thryothorus ludovicianus", "Carolina Wren"), ("Poecile atricapillus", "Black-capped Chickadee"),
                 ("Cyanocitta cristata", "Blue Jay"), ("Turdus migratorius", "American Robin")]},
    {"title": "3 · Photo + song", "kinds": ["photo", "song"], "split": True,
     "species": [("Dryobates pubescens", "Downy Woodpecker"), ("Melanerpes carolinus", "Red-bellied Woodpecker"),
                 ("Cyanocitta cristata", "Blue Jay")]},
]

# ---------------------------------------------------------------- start the app
open(f"{WORK}/app.py", "wb").write(base64.b64decode(APP_B64))
env = dict(os.environ)
env["BURDS_DATA"] = os.path.dirname(glob.glob("/kaggle/input/**/demo_index.npz", recursive=True)[0])
env["BURDS_TAXA"] = [p for p in glob.glob("/kaggle/input/**/taxa.csv", recursive=True) if "demo-data" in p][0]
cfg = glob.glob("/kaggle/input/**/embeddinggemma*/**/config.json", recursive=True)
if cfg:
    env["BURDS_MODEL"] = os.path.dirname(sorted(cfg, key=len)[0])
env.pop("BURDS_SHARE", None)
app = subprocess.Popen([sys.executable, f"{WORK}/app.py"], env=env, cwd=WORK)
URL = "http://127.0.0.1:7860"
for _ in range(600):
    try:
        if S.get(URL, timeout=2).ok:
            break
    except requests.RequestException:
        pass
    assert app.poll() is None, "app exited"
    time.sleep(1)
log("app is up")


def predict(photo=None, song=None):
    def up(f):
        r = S.post(f"{URL}/gradio_api/upload", files={"files": open(f, "rb")}, timeout=60)
        return {"path": r.json()[0], "meta": {"_type": "gradio.FileData"}}
    data = [up(photo) if photo else None, up(song) if song else None]
    eid = S.post(f"{URL}/gradio_api/call/identify", json={"data": data}, timeout=60).json()["event_id"]
    txt = S.get(f"{URL}/gradio_api/call/identify/{eid}", timeout=300).text
    out = json.loads([x for x in txt.splitlines() if x.startswith("data:")][-1][5:])
    return [o.get("label") if isinstance(o, dict) else None for o in out[:3]]  # fused, photo-only, song-only


for sc in SCENES:
    sc["tries"] = 0
    for sci, common in sc["species"]:
        ok = lambda lab: bool(lab) and lab.startswith(common)  # noqa: E731
        chosen = {}
        for kind in sc["kinds"]:
            first = None
            for o, m in candidates(sci, common, kind):
                media = fetch_media(o, m, kind)
                if not media:
                    continue
                first = first or media
                sc["tries"] += 1
                fused = predict(**{kind: media["file"]})[0]
                log(f"  {sc['title']} {common} {kind} {media['url']}: {fused}")
                if ok(fused):
                    chosen[kind] = media
                    break
            if kind not in chosen and kind == "song" and "photo" in chosen and first:
                chosen[kind] = first  # photo + song: the photo can carry the fused answer
        if len(chosen) == len(sc["kinds"]):
            if ok(predict(chosen.get("photo", {}).get("file"), chosen.get("song", {}).get("file"))[0]):
                sc["species_name"] = common
                sc.update({f"{k}_media": v for k, v in chosen.items()})
                break
    assert "species_name" in sc, f"no working example for {sc['title']}"
    log(f"{sc['title']}: {sc['species_name']} after {sc['tries']} tries")

# ---------------------------------------------------------------- drive it
from playwright.sync_api import sync_playwright  # noqa: E402

W, H = 1280, 900
BANNER = """(t) => { let b = document.getElementById('burds-banner');
  if (!b) { b = document.createElement('div'); b.id = 'burds-banner'; document.body.appendChild(b);
    Object.assign(b.style, {position: 'fixed', left: '0', right: '0', bottom: '0', zIndex: 9999,
      padding: '10px 18px', background: 'rgba(17,24,39,.92)', color: '#fff',
      font: '600 18px/1.35 system-ui, sans-serif'}); }
  b.innerHTML = t; }"""


def run_scene(page, sc, shot=None):
    page.goto(URL)
    page.locator("#identify").wait_for()
    credit = " · ".join(f"<span style='font-weight:400;font-size:14px'>{k} {sc[m]['author']} "
                        f"via iNaturalist</span>"
                        for k, m in (("photo", "photo_media"), ("song", "song_media")) if m in sc)
    page.evaluate(BANNER, f"{sc['title']}: a {sc['species_name']} the model has never seen<br>{credit}")
    page.wait_for_timeout(1500)
    if "photo_media" in sc:
        page.locator("#photo input[type=file]").set_input_files(sc["photo_media"]["file"])
        page.wait_for_timeout(1500)
    if "song_media" in sc:
        page.locator("#song input[type=file]").set_input_files(sc["song_media"]["file"])
        page.wait_for_timeout(2000)
    page.locator("#identify").click()
    page.wait_for_function("() => (document.querySelector('#best')?.innerText || '').includes('%')",
                           timeout=120000)
    page.locator("#best").scroll_into_view_if_needed()
    page.wait_for_timeout(2500)
    if sc.get("split"):
        page.locator("#split").get_by_text("Photo only / song only").click()
        page.wait_for_timeout(1000)
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(3000)
    best = page.locator("#best").inner_text()
    if shot:
        page.screenshot(path=shot)
    return best


with sync_playwright() as p:
    br = p.chromium.launch()
    warm = br.new_page(viewport={"width": W, "height": H})  # first call loads CUDA kernels; keep it off camera
    run_scene(warm, SCENES[0])
    warm.close()
    ctx = br.new_context(viewport={"width": W, "height": H}, record_video_dir=WORK,
                         record_video_size={"width": W, "height": H}, device_scale_factor=1)
    page = ctx.new_page()
    for i, sc in enumerate(SCENES, 1):
        sc["prediction"] = run_scene(page, sc, f"{OUT}/scene{i}.png")
        log(f"{sc['title']} -> {sc['prediction'].splitlines()[:3]}")
    video = page.video.path()
    ctx.close()
    br.close()
app.terminate()

sh(f"ffmpeg -y -loglevel error -i {video} -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart {OUT}/burds_demo.mp4")
sh(f"ffmpeg -y -loglevel error -i {video} -vf 'fps=8,scale=900:-1:flags=lanczos,split[a][b];"
   f"[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer' {OUT}/burds_demo.gif")
for sc in SCENES:
    for m in ("photo_media", "song_media"):
        if m in sc:
            sc[m]["file"] = os.path.basename(sc[m]["file"])
json.dump(SCENES, open(f"{OUT}/media.json", "w"), indent=2)
sh(f"ls -la {OUT}")
log("done")
