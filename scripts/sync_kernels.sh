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
