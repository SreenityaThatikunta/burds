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
APP_B64 = "IiIiCmJ1cmRzOiBpZGVudGlmeSBhIE5vcnRoIEFtZXJpY2FuIGJpcmQgZnJvbSBhIHBob3RvLCBhIHNvbmcsIG9yIGJvdGguCgpObyB0cmFpbmluZyBvbiB0aGUgbW9kZWwgaXRzZWxmOiBFbWJlZGRpbmdHZW1tYSAyIHR1cm5zIHRoZSBwaG90byBhbmQgdGhlIHNvbmcgaW50byB2ZWN0b3JzLCBhbmQKdHdvIHNtYWxsIGxpbmVhciBjbGFzc2lmaWVycyAoNjAgc3BlY2llcywgdHJhaW5lZCBvbiBvcGVubHkgbGljZW5zZWQgaU5hdHVyYWxpc3QgcGhvdG9zIGFuZAppTmF0dXJhbGlzdCArIHhlbm8tY2FudG8gcmVjb3JkaW5ncykgcmVhZCB0aG9zZSB2ZWN0b3JzLiBQaG90byBhbmQgc29uZyBwcmVkaWN0aW9ucyBhcmUgY29tYmluZWQgYnkgY29uZmlkZW5jZS4KIiIiCmltcG9ydCBjc3YKaW1wb3J0IG9zCmltcG9ydCB0aW1lCgppbXBvcnQgZ3JhZGlvIGFzIGdyCmltcG9ydCBudW1weSBhcyBucAppbXBvcnQgdG9yY2gKZnJvbSBQSUwgaW1wb3J0IEltYWdlCmZyb20gc2VudGVuY2VfdHJhbnNmb3JtZXJzIGltcG9ydCBTZW50ZW5jZVRyYW5zZm9ybWVyCgojIC0tLS0gYXBwL2J1cmRzX2NvcmUucHkgKGlubGluZWQpIC0tLS0KIiIiClB1cmUgbnVtcHkvc2NpcHkgY29yZSBvZiBidXJkczogYXVkaW8gcHJlcHJvY2Vzc2luZyBhbmQgc2NvcmUgZnVzaW9uLiBObyBtb2RlbCwgbm8gVUksIHNvIGl0IGlzCmZhc3QgdG8gdGVzdCAodGVzdHMvKSBhbmQgc2hhcmVkIGJ5IHRoZSBhcHAgYW5kIHRoZSBLYWdnbGUga2VybmVscy4KClRoZSBhdWRpbyBmdW5jdGlvbnMgbXVzdCBzdGF5IGJ5dGUtaWRlbnRpY2FsIHRvIHRoZSBjb3BpZXMgaW4ga2FnZ2xlL2RlbW9fZW1iZWQvZGVtb19lbWJlZC5weQooS2FnZ2xlIGtlcm5lbHMgYXJlIHNpbmdsZSBmaWxlcyk7IHRlc3RzL3Rlc3RfY29uc2lzdGVuY3kucHkgY2hlY2tzIHRoaXMuCiIiIgppbXBvcnQgbnVtcHkgYXMgbnAKClNSID0gMTYwMDAgICMgRW1iZWRkaW5nR2VtbWEgMiBhdWRpbyByYXRlOiBpdCBoZWFycyAwLTgga0h6CgoKZGVmIHJlc2FtcGxlKHgsIHNyX2luLCBzcl9vdXQpOgogICAgZnJvbSBtYXRoIGltcG9ydCBnY2QKCiAgICBmcm9tIHNjaXB5LnNpZ25hbCBpbXBvcnQgcmVzYW1wbGVfcG9seQogICAgc3JfaW4sIHNyX291dCA9IGludChyb3VuZChzcl9pbikpLCBpbnQocm91bmQoc3Jfb3V0KSkKICAgIGcgPSBnY2Qoc3JfaW4sIHNyX291dCkKICAgIHJldHVybiByZXNhbXBsZV9wb2x5KHgsIHNyX291dCAvLyBnLCBzcl9pbiAvLyBnKS5hc3R5cGUobnAuZmxvYXQzMikKCgpkZWYgbG91ZGVzdCh4LCBzciwgc2Vjcyk6CiAgICBmcm9tIHNjaXB5LnNpZ25hbCBpbXBvcnQgYnV0dGVyLCBzb3NmaWx0CiAgICBuID0gaW50KHNlY3MgKiBzcikKICAgIGlmIGxlbih4KSA8PSBuOgogICAgICAgIHJldHVybiAwCiAgICBocCA9IHNvc2ZpbHQoYnV0dGVyKDQsIDEwMDAsICJoaWdocGFzcyIsIGZzPXNyLCBvdXRwdXQ9InNvcyIpLCB4KQogICAgYyA9IG5wLmNvbmNhdGVuYXRlKFtbMC4wXSwgbnAuY3Vtc3VtKGhwLmFzdHlwZShucC5mbG9hdDY0KSAqKiAyKV0pCiAgICByZXR1cm4gaW50KG5wLmFyZ21heChjW246XSAtIGNbOi1uXSkpCgoKZGVmIHJtcyh4KToKICAgIHJldHVybiBmbG9hdChucC5zcXJ0KG5wLm1lYW4oeC5hc3R5cGUobnAuZmxvYXQ2NCkgKiogMikpICsgMWUtOSkKCgpkZWYgYWRkX2JhY2tncm91bmQoeCwgc3IsIG5vaXNlLCBzcl9uLCBzbnJfZGIpOgogICAgIiIiTWl4IGluIHRoZSBxdWlldGVzdCBzdHJldGNoIG9mIGFub3RoZXIgcmVjb3JkaW5nICh3aW5kLCB0cmFmZmljLCBvdGhlciBiaXJkcywgaGlzcykgYXQgc25yX2RiLiIiIgogICAgaWYgc3JfbiAhPSBzcjoKICAgICAgICBub2lzZSA9IHJlc2FtcGxlKG5vaXNlLCBzcl9uLCBzcikKICAgIG4gPSBtaW4obGVuKHgpLCBsZW4obm9pc2UpKQogICAgYyA9IG5wLmNvbmNhdGVuYXRlKFtbMC4wXSwgbnAuY3Vtc3VtKG5vaXNlLmFzdHlwZShucC5mbG9hdDY0KSAqKiAyKV0pCiAgICBzMCA9IGludChucC5hcmdtaW4oY1tuOl0gLSBjWzotbl0pKSBpZiBsZW4obm9pc2UpID4gbiBlbHNlIDAKICAgIHNlZyA9IG5wLnJlc2l6ZShub2lzZVtzMDpzMCArIG5dLCBsZW4oeCkpICAjIHRpbGUgaWYgdGhlIGJhY2tncm91bmQgaXMgc2hvcnRlciB0aGFuIHRoZSBjbGlwCiAgICByZXR1cm4gKHggKyBzZWcgKiAocm1zKHgpIC8gcm1zKHNlZykpICogMTAgKiogKC1zbnJfZGIgLyAyMCkpLmFzdHlwZShucC5mbG9hdDMyKQoKCmRlZiBhdWRpb192YXJpYW50KHgsIHNyLCBraW5kKToKICAgICIiImtpbmQ6IFtzbG93Ml9ddHJpbTxOPi4gSGFsZiBzcGVlZCByZWFkcyB0aGUgc2FtcGxlcyBhdCBzci8yLCBzbyBmcmVxdWVuY3kgZiBsYW5kcyBhdCBmLzIuIiIiCiAgICByYXRlID0gc3IgLyAyIGlmIGtpbmQuc3RhcnRzd2l0aCgic2xvdzIiKSBlbHNlIHNyCiAgICBuID0gZmxvYXQoa2luZC5zcGxpdCgidHJpbSIpWzFdKQogICAgcyA9IGxvdWRlc3QoeCwgc3IsIG4pCiAgICByZXR1cm4gcmVzYW1wbGUoeFtzOnMgKyBpbnQobiAqIHNyKV0sIHJhdGUsIFNSKQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBzY29yZXMKZGVmIHNvZnRtYXgoeik6CiAgICB6ID0geiAtIHoubWF4KCkKICAgIGUgPSBucC5leHAoeikKICAgIHJldHVybiBlIC8gZS5zdW0oKQoKCmRlZiBibG9ja3Nfbm9ybSh2LCBkPTc2OCk6CiAgICAiIiJMMi1ub3JtYWxpc2UgZWFjaCA3NjgtZCBibG9jayBvZiBhIGNvbmNhdGVuYXRlZCBlbWJlZGRpbmcgKG9uZSBibG9jayBwZXIgYXVkaW8gdmVyc2lvbikuIiIiCiAgICB2ID0gdi5yZXNoYXBlKC0xLCBkKQogICAgcmV0dXJuICh2IC8gbnAubGluYWxnLm5vcm0odiwgYXhpcz0xLCBrZWVwZGltcz1UcnVlKS5jbGlwKDFlLTEyKSkucmVzaGFwZSgtMSkKCgpkZWYgZnVzZShwaSwgcGEsIGdhbW1hKToKICAgICIiIkNvbmZpZGVuY2UgZnVzaW9uOiBhdmVyYWdlIHBob3RvIGFuZCBzb25nIGxvZy1wcm9icywgZWFjaCB3ZWlnaHRlZCBieSBpdHMgdG9wIHByb2JhYmlsaXR5ICoqIGdhbW1hLiIiIgogICAgYSwgYiA9IHBpLm1heCgpICoqIGdhbW1hLCBwYS5tYXgoKSAqKiBnYW1tYQogICAgcmV0dXJuIHNvZnRtYXgoKGEgKiBucC5sb2cocGkgKyAxZS05KSArIGIgKiBucC5sb2cocGEgKyAxZS05KSkgLyAoYSArIGIpKQoKCmRlZiB0b3BfayhwLCBuYW1lcywgaz01KToKICAgIHJldHVybiB7bmFtZXNbaV06IGZsb2F0KHBbaV0pIGZvciBpIGluIG5wLmFyZ3NvcnQoLXApWzprXX0KCiMgLS0tLSBlbmQgYnVyZHNfY29yZSAtLS0tCgpIRVJFID0gb3MuZW52aXJvbi5nZXQoIkJVUkRTX0RBVEEiKSBvciBvcy5wYXRoLmRpcm5hbWUob3MucGF0aC5hYnNwYXRoKF9fZmlsZV9fKSkKSURYID0gbnAubG9hZChvcy5wYXRoLmpvaW4oSEVSRSwgImRlbW9faW5kZXgubnB6IikpCkNPTU1PTiwgU0NJID0gSURYWyJjb21tb24iXS50b2xpc3QoKSwgSURYWyJzY2kiXS50b2xpc3QoKQpOQU1FUyA9IFtmIntjfSAoe3N9KSIgZm9yIGMsIHMgaW4gemlwKENPTU1PTiwgU0NJLCBzdHJpY3Q9VHJ1ZSldCndpdGggb3Blbihvcy5lbnZpcm9uLmdldCgiQlVSRFNfVEFYQSIpIG9yIG9zLnBhdGguam9pbihIRVJFLCAidGF4YS5jc3YiKSkgYXMgZjoKICAgIENPREVTID0gW3JbInNwZWNpZXNfY29kZSJdIGZvciByIGluIGNzdi5EaWN0UmVhZGVyKGYpXQpXX0lNRywgQl9JTUcsIFdfQVVELCBCX0FVRCA9IElEWFsiV19pbWciXSwgSURYWyJiX2ltZyJdLCBJRFhbIldfYXVkIl0sIElEWFsiYl9hdWQiXQpHQU1NQSwgQlVER0VUID0gZmxvYXQoSURYWyJnYW1tYSJdKSwgaW50KElEWFsiYnVkZ2V0Il0pCkFVRElPX0tJTkRTID0gSURYWyJhdWRpb19raW5kcyJdLnRvbGlzdCgpCgp0b3JjaC5zZXRfbnVtX3RocmVhZHMob3MuY3B1X2NvdW50KCkgb3IgMikKREVWSUNFID0gImN1ZGEiIGlmIHRvcmNoLmN1ZGEuaXNfYXZhaWxhYmxlKCkgZWxzZSAiY3B1IgpNT0RFTCA9IFNlbnRlbmNlVHJhbnNmb3JtZXIob3MuZW52aXJvbi5nZXQoIkJVUkRTX01PREVMIiwgImdvb2dsZS9lbWJlZGRpbmdnZW1tYS0yIiksIGRldmljZT1ERVZJQ0UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBtb2RlbF9rd2FyZ3M9eyJ0b3JjaF9kdHlwZSI6IHRvcmNoLmZsb2F0MzJ9KQoKCmRlZiBsb2FkX2F1ZGlvKHBhdGgpOgogICAgaW1wb3J0IGxpYnJvc2EKICAgIHgsIHNyID0gbGlicm9zYS5sb2FkKHBhdGgsIHNyPU5vbmUsIG1vbm89VHJ1ZSwgZHVyYXRpb249MTIwKQogICAgcmV0dXJuIG5wLmFzYXJyYXkoeCwgbnAuZmxvYXQzMiksIHNyCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIHByZWRpY3Rpb24KZGVmIHBob3RvX3Byb2JzKGltZyk6CiAgICBlID0gTU9ERUwuZW5jb2RlKFt7ImltYWdlIjogaW1nLmNvbnZlcnQoIlJHQiIpfV0sIG5vcm1hbGl6ZV9lbWJlZGRpbmdzPVRydWUsIGNvbnZlcnRfdG9fbnVtcHk9VHJ1ZSwKICAgICAgICAgICAgICAgICAgICAgcHJvY2Vzc2luZ19rd2FyZ3M9eyJpbWFnZSI6IHsibWF4X3NvZnRfdG9rZW5zIjogQlVER0VUfX0pWzBdCiAgICByZXR1cm4gc29mdG1heChXX0lNRyBAIGUgKyBCX0lNRykKCgpkZWYgc29uZ19wcm9icyhwYXRoKToKICAgIHgsIHNyID0gbG9hZF9hdWRpbyhwYXRoKQogICAgaWYgbGVuKHgpIDwgc3I6CiAgICAgICAgcmFpc2UgZ3IuRXJyb3IoIlRoZSByZWNvcmRpbmcgaXMgc2hvcnRlciB0aGFuIDEgc2Vjb25kLiIpCiAgICBlID0gTU9ERUwuZW5jb2RlKFt7ImF1ZGlvIjogYXVkaW9fdmFyaWFudCh4LCBzciwgayl9IGZvciBrIGluIEFVRElPX0tJTkRTXSwKICAgICAgICAgICAgICAgICAgICAgbm9ybWFsaXplX2VtYmVkZGluZ3M9VHJ1ZSwgY29udmVydF90b19udW1weT1UcnVlKQogICAgcmV0dXJuIHNvZnRtYXgoV19BVUQgQCBibG9ja3Nfbm9ybShlLnJlc2hhcGUoLTEpKSArIEJfQVVEKQoKCmRlZiB0b3AocCwgaz01KToKICAgIHJldHVybiB0b3BfayhwLCBOQU1FUywgaykKCgpkZWYgaWRlbnRpZnkoaW1nLCBhdWRpbyk6CiAgICBpZiBpbWcgaXMgTm9uZSBhbmQgYXVkaW8gaXMgTm9uZToKICAgICAgICByYWlzZSBnci5FcnJvcigiQWRkIGEgcGhvdG8sIGEgcmVjb3JkaW5nLCBvciBib3RoLiIpCiAgICBwaSA9IHBob3RvX3Byb2JzKGltZykgaWYgaW1nIGlzIG5vdCBOb25lIGVsc2UgTm9uZQogICAgcGEgPSBzb25nX3Byb2JzKGF1ZGlvKSBpZiBhdWRpbyBpcyBub3QgTm9uZSBlbHNlIE5vbmUKICAgIHAgPSBmdXNlKHBpLCBwYSwgR0FNTUEpIGlmIHBpIGlzIG5vdCBOb25lIGFuZCBwYSBpcyBub3QgTm9uZSBlbHNlIChwaSBpZiBwaSBpcyBub3QgTm9uZSBlbHNlIHBhKQogICAgYmVzdCA9IGludChwLmFyZ21heCgpKQogICAgdGh1bWIgPSBvcy5wYXRoLmpvaW4oSEVSRSwgInRodW1icyIsIGYie0NPREVTW2Jlc3RdfS5qcGciKQogICAgcmV0dXJuICh0b3AocCksIHRvcChwaSkgaWYgcGkgaXMgbm90IE5vbmUgZWxzZSBOb25lLCB0b3AocGEpIGlmIHBhIGlzIG5vdCBOb25lIGVsc2UgTm9uZSwKICAgICAgICAgICAgSW1hZ2Uub3Blbih0aHVtYikuY29udmVydCgiUkdCIikgaWYgb3MucGF0aC5leGlzdHModGh1bWIpIGVsc2UgTm9uZSkKCgp3aXRoIGdyLkJsb2Nrcyh0aXRsZT0iYnVyZHM6IGJpcmQgSUQgZnJvbSBwaG90byArIHNvbmciKSBhcyBkZW1vOgogICAgZ3IuTWFya2Rvd24oCiAgICAgICAgIiMgYnVyZHNcbklkZW50aWZ5IG9uZSBvZiAqKjYwIE5vcnRoIEFtZXJpY2FuIGJpcmRzKiogZnJvbSBhIHBob3RvLCBhIHNvbmcsIG9yIGJvdGguICIKICAgICAgICAiT25lIG1vZGVsIChbRW1iZWRkaW5nR2VtbWEgMl0oaHR0cHM6Ly9odWdnaW5nZmFjZS5jby9nb29nbGUvZW1iZWRkaW5nZ2VtbWEtMikpIGVtYmVkcyBib3RoLCAiCiAgICAgICAgIndpdGggbm8gZmluZS10dW5pbmc6IHNtYWxsIGxpbmVhciBjbGFzc2lmaWVycyB0cmFpbmVkIG9uIG9wZW5seSBsaWNlbnNlZCBpTmF0dXJhbGlzdCBwaG90b3MgYW5kICIKICAgICAgICAiaU5hdHVyYWxpc3QgKyB4ZW5vLWNhbnRvIHJlY29yZGluZ3MgZG8gdGhlIHJlc3QuIFtIb3cgaXQgd29ya3MgYW5kIHJlc3VsdHNdKGh0dHBzOi8vZ2l0aHViLmNvbS9TcmVlbml0eWFUaGF0aWt1bnRhL2J1cmRzKSIpCiAgICB3aXRoIGdyLlJvdygpOgogICAgICAgIHdpdGggZ3IuQ29sdW1uKCk6CiAgICAgICAgICAgIGltZyA9IGdyLkltYWdlKHR5cGU9InBpbCIsIGxhYmVsPSJQaG90byIsIHNvdXJjZXM9WyJ1cGxvYWQiLCAid2ViY2FtIiwgImNsaXBib2FyZCJdLCBlbGVtX2lkPSJwaG90byIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgIGhlaWdodD0zMDApCiAgICAgICAgICAgIGF1ZCA9IGdyLkF1ZGlvKHR5cGU9ImZpbGVwYXRoIiwgbGFiZWw9IlNvbmcgb3IgY2FsbCIsIHNvdXJjZXM9WyJ1cGxvYWQiLCAibWljcm9waG9uZSJdLCBlbGVtX2lkPSJzb25nIikKICAgICAgICAgICAgYnRuID0gZ3IuQnV0dG9uKCJJZGVudGlmeSIsIHZhcmlhbnQ9InByaW1hcnkiLCBlbGVtX2lkPSJpZGVudGlmeSIpCiAgICAgICAgd2l0aCBnci5Db2x1bW4oKToKICAgICAgICAgICAgb3V0ID0gZ3IuTGFiZWwobnVtX3RvcF9jbGFzc2VzPTUsIGxhYmVsPSJCZXN0IGd1ZXNzIiwgZWxlbV9pZD0iYmVzdCIpCiAgICAgICAgICAgIHJlZiA9IGdyLkltYWdlKGxhYmVsPSJSZWZlcmVuY2UgcGhvdG8gKGlOYXR1cmFsaXN0LCBzZWUgY3JlZGl0cykiLCBpbnRlcmFjdGl2ZT1GYWxzZSwgaGVpZ2h0PTIyMCkKICAgICAgICAgICAgd2l0aCBnci5BY2NvcmRpb24oIlBob3RvIG9ubHkgLyBzb25nIG9ubHkiLCBvcGVuPUZhbHNlLCBlbGVtX2lkPSJzcGxpdCIpOgogICAgICAgICAgICAgICAgb3V0X2kgPSBnci5MYWJlbChudW1fdG9wX2NsYXNzZXM9NSwgbGFiZWw9IkZyb20gdGhlIHBob3RvIikKICAgICAgICAgICAgICAgIG91dF9hID0gZ3IuTGFiZWwobnVtX3RvcF9jbGFzc2VzPTUsIGxhYmVsPSJGcm9tIHRoZSBzb25nIikKICAgIGdyLk1hcmtkb3duKGYiUnVubmluZyBvbiB7REVWSUNFLnVwcGVyKCl9LiBTcGVjaWVzIGxpc3QgYW5kIG1lZGlhIGNyZWRpdHM6ICIKICAgICAgICAgICAgICAgICJgdGF4YS5jc3ZgLCBgY3JlZGl0cy5jc3ZgIGluIHRoaXMgU3BhY2UuIikKICAgIGJ0bi5jbGljayhpZGVudGlmeSwgW2ltZywgYXVkXSwgW291dCwgb3V0X2ksIG91dF9hLCByZWZdKQoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIGlmIG9zLmVudmlyb24uZ2V0KCJCVVJEU19TSEFSRSIpOiAgIyB0ZW1wb3JhcnkgcHVibGljIGxpbmsgKGUuZy4gZnJvbSBhIEthZ2dsZSBHUFUgcnVuKQogICAgICAgIGRlbW8ubGF1bmNoKHNoYXJlPVRydWUsIHByZXZlbnRfdGhyZWFkX2xvY2s9VHJ1ZSkKICAgICAgICBwcmludChmIkJVUkRTX0xJVkUgc2hhcmUgbGluayBhYm92ZTsgc3RheWluZyB1cCBmb3Ige29zLmVudmlyb24uZ2V0KCdCVVJEU19IT1VSUycsICczJyl9IGgiLCBmbHVzaD1UcnVlKQogICAgICAgIHRpbWUuc2xlZXAoZmxvYXQob3MuZW52aXJvbi5nZXQoIkJVUkRTX0hPVVJTIiwgIjMiKSkgKiAzNjAwKQogICAgZWxzZToKICAgICAgICBkZW1vLmxhdW5jaCgpCg=="
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
