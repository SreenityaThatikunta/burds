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

# demo_live = Kaggle wrapper + the exact app that ships to a Space
cat kaggle/demo_live/header.py app/app.py > kaggle/demo_live/demo_live.py
echo "synced kaggle/demo_live/demo_live.py"

# demo_video = recording script with the exact app embedded (base64, so no quoting issues)
python3 - <<'PY'
import base64
t = open("kaggle/demo_video/video_template.py").read()
b = base64.b64encode(open("app/app.py", "rb").read()).decode()
assert "@@APP_B64@@" in t
open("kaggle/demo_video/demo_video.py", "w").write(t.replace("@@APP_B64@@", b))
PY
echo "synced kaggle/demo_video/demo_video.py"
