#!/usr/bin/env bash
# Kaggle script kernels upload a single code file, so experiments that share the embed
# implementation get a copy of kaggle/ssw60_embed/ssw60_embed.py with EXPERIMENT changed.
# Run this after editing ssw60_embed.py, before `kaggle kernels push`.
set -euo pipefail
cd "$(dirname "$0")/.."
src=kaggle/ssw60_embed/ssw60_embed.py

copy() {  # copy <experiment> <dest>
  sed "s/^EXPERIMENT = \"variants\"$/EXPERIMENT = \"$1\"/" "$src" > "$2"
  grep -q "^EXPERIMENT = \"$1\"$" "$2" || { echo "failed to set EXPERIMENT in $2" >&2; exit 1; }
  echo "synced $2 ($1)"
}

copy audiolen kaggle/ssw60_audiolen/ssw60_audiolen.py

# Single-file app for Kaggle: app/burds_core.py inlined in place of its import.
python3 - <<'PY'
import base64, re
core = open("app/burds_core.py").read()
app = open("app/app.py").read()
line = re.search(r"^from burds_core import .*$", app, re.M)
assert line, "app.py no longer imports burds_core"
single = app.replace(line.group(0), "# ---- app/burds_core.py (inlined) ----\n" + core + "\n# ---- end burds_core ----")

# demo_live = Kaggle wrapper + the exact app that ships to a Space
open("kaggle/demo_live/demo_live.py", "w").write(open("kaggle/demo_live/header.py").read() + single)

# demo_video / demo_bench = scripts with the exact app embedded (base64, so no quoting issues)
b = base64.b64encode(single.encode()).decode()
for t_path, out in (("kaggle/demo_video/video_template.py", "kaggle/demo_video/demo_video.py"),
                    ("kaggle/demo_bench/bench_template.py", "kaggle/demo_bench/demo_bench.py")):
    t = open(t_path).read()
    assert "@@APP_B64@@" in t, t_path
    open(out, "w").write(t.replace("@@APP_B64@@", b))
PY
echo "synced kaggle/demo_live, kaggle/demo_video, kaggle/demo_bench"
