"""
burds: identify a North American bird from a photo, a song, or both.

No training on the model itself: EmbeddingGemma 2 turns the photo and the song into vectors, and
two small linear classifiers (60 species, trained on openly licensed iNaturalist photos and
iNaturalist + xeno-canto recordings) read those vectors. Photo and song predictions are combined by confidence.
"""
import csv
import os
import time

import gradio as gr
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from burds_core import audio_variant, blocks_norm, fuse, softmax, top_k  # noqa: E402  (inlined on Kaggle)

HERE = os.environ.get("BURDS_DATA") or os.path.dirname(os.path.abspath(__file__))
IDX = np.load(os.path.join(HERE, "demo_index.npz"))
COMMON, SCI = IDX["common"].tolist(), IDX["sci"].tolist()
NAMES = [f"{c} ({s})" for c, s in zip(COMMON, SCI, strict=True)]
with open(os.environ.get("BURDS_TAXA") or os.path.join(HERE, "taxa.csv")) as f:
    CODES = [r["species_code"] for r in csv.DictReader(f)]
W_IMG, B_IMG, W_AUD, B_AUD = IDX["W_img"], IDX["b_img"], IDX["W_aud"], IDX["b_aud"]
GAMMA, BUDGET = float(IDX["gamma"]), int(IDX["budget"])
AUDIO_KINDS = IDX["audio_kinds"].tolist()

torch.set_num_threads(os.cpu_count() or 2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer(os.environ.get("BURDS_MODEL", "google/embeddinggemma-2"), device=DEVICE,
                            model_kwargs={"torch_dtype": torch.float32})


def load_audio(path):
    import librosa
    x, sr = librosa.load(path, sr=None, mono=True, duration=120)
    return np.asarray(x, np.float32), sr


# ---------------------------------------------------------------- prediction
def photo_probs(img):
    e = MODEL.encode([{"image": img.convert("RGB")}], normalize_embeddings=True, convert_to_numpy=True,
                     processing_kwargs={"image": {"max_soft_tokens": BUDGET}})[0]
    return softmax(W_IMG @ e + B_IMG)


def song_probs(path):
    x, sr = load_audio(path)
    if len(x) < sr:
        raise gr.Error("The recording is shorter than 1 second.")
    e = MODEL.encode([{"audio": audio_variant(x, sr, k)} for k in AUDIO_KINDS],
                     normalize_embeddings=True, convert_to_numpy=True)
    return softmax(W_AUD @ blocks_norm(e.reshape(-1)) + B_AUD)


def top(p, k=5):
    return top_k(p, NAMES, k)


def identify(img, audio):
    if img is None and audio is None:
        raise gr.Error("Add a photo, a recording, or both.")
    pi = photo_probs(img) if img is not None else None
    pa = song_probs(audio) if audio is not None else None
    p = fuse(pi, pa, GAMMA) if pi is not None and pa is not None else (pi if pi is not None else pa)
    best = int(p.argmax())
    thumb = os.path.join(HERE, "thumbs", f"{CODES[best]}.jpg")
    return (top(p), top(pi) if pi is not None else None, top(pa) if pa is not None else None,
            Image.open(thumb).convert("RGB") if os.path.exists(thumb) else None)


with gr.Blocks(title="burds: bird ID from photo + song") as demo:
    gr.Markdown(
        "# burds\nIdentify one of **60 North American birds** from a photo, a song, or both. "
        "One model ([EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)) embeds both, "
        "with no fine-tuning: small linear classifiers trained on openly licensed iNaturalist photos and "
        "iNaturalist + xeno-canto recordings do the rest. [How it works and results](https://github.com/SreenityaThatikunta/burds)")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Photo", sources=["upload", "webcam", "clipboard"], elem_id="photo",
                           height=300)
            aud = gr.Audio(type="filepath", label="Song or call", sources=["upload", "microphone"], elem_id="song")
            btn = gr.Button("Identify", variant="primary", elem_id="identify")
        with gr.Column():
            out = gr.Label(num_top_classes=5, label="Best guess", elem_id="best")
            ref = gr.Image(label="Reference photo (iNaturalist, see credits)", interactive=False, height=220)
            with gr.Accordion("Photo only / song only", open=False, elem_id="split"):
                out_i = gr.Label(num_top_classes=5, label="From the photo")
                out_a = gr.Label(num_top_classes=5, label="From the song")
    gr.Markdown(f"Running on {DEVICE.upper()}. Species list and media credits: "
                "`taxa.csv`, `credits.csv` in this Space.")
    btn.click(identify, [img, aud], [out, out_i, out_a, ref])

if __name__ == "__main__":
    if os.environ.get("BURDS_SHARE"):  # temporary public link (e.g. from a Kaggle GPU run)
        demo.launch(share=True, prevent_thread_lock=True)
        print(f"BURDS_LIVE share link above; staying up for {os.environ.get('BURDS_HOURS', '3')} h", flush=True)
        time.sleep(float(os.environ.get("BURDS_HOURS", "3")) * 3600)
    else:
        demo.launch()
