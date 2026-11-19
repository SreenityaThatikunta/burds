"""The app must preprocess audio exactly as the Kaggle kernel that trained its classifiers did,
and the shipped classifier file must match the app's expectations."""
import ast
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHARED = ("resample", "loudest", "rms", "add_background", "audio_variant")


def functions(path):
    src = path.read_text()
    return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


def test_app_and_training_kernel_share_audio_code():
    core = functions(ROOT / "app" / "burds_core.py")
    kernel = functions(ROOT / "kaggle" / "demo_embed" / "demo_embed.py")
    for name in SHARED:
        assert core[name] == kernel[name], f"{name} differs between app/burds_core.py and demo_embed.py"


def test_generated_kaggle_scripts_are_in_sync():
    # scripts/sync_kernels.sh inlines burds_core into the app for single-file Kaggle kernels
    core = (ROOT / "app" / "burds_core.py").read_text()
    live = (ROOT / "kaggle" / "demo_live" / "demo_live.py").read_text()
    assert core in live, "run scripts/sync_kernels.sh"


def test_demo_index_matches_taxa():
    idx = np.load(ROOT / "app" / "demo_index.npz")
    taxa = list(csv.DictReader(open(ROOT / "app" / "taxa.csv")))
    k = len(taxa)
    assert k == 60
    assert idx["W_img"].shape == (k, 768)
    assert idx["W_aud"].shape == (k, 768 * len(idx["audio_kinds"]))
    assert [t["common_name"] for t in taxa] == idx["common"].tolist()
    assert all((ROOT / "app" / "thumbs" / f"{t['species_code']}.jpg").exists() for t in taxa)
