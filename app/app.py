"""
burds: identify a North American bird from a photo, a song, or both.

No training on the model itself: EmbeddingGemma 2 turns the photo and the song into vectors, and
two small linear classifiers (60 species, trained on openly licensed iNaturalist photos and
xeno-canto recordings) read those vectors. Photo and song predictions are combined by confidence.
"""
import csv
import os

import gradio as gr
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
IDX = np.load(os.path.join(HERE, "demo_index.npz"))
COMMON, SCI = IDX["common"].tolist(), IDX["sci"].tolist()
CODES = [r["species_code"] for r in csv.DictReader(open(os.path.join(HERE, "taxa.csv")))]
W_IMG, B_IMG, W_AUD, B_AUD = IDX["W_img"], IDX["b_img"], IDX["W_aud"], IDX["b_aud"]
GAMMA, BUDGET = float(IDX["gamma"]), int(IDX["budget"])
AUDIO_KINDS = IDX["audio_kinds"].tolist()

torch.set_num_threads(os.cpu_count() or 2)
MODEL = SentenceTransformer(os.environ.get("BURDS_MODEL", "google/embeddinggemma-2"), device="cpu",
                            model_kwargs={"torch_dtype": torch.float32})


# ---------------------------------------------------------------- audio (identical to the Kaggle recipe)
def resample(x, sr_in, sr_out):
    from math import gcd
    from scipy.signal import resample_poly
    sr_in, sr_out = int(round(sr_in)), int(round(sr_out))
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def loudest(x, sr, secs):
    from scipy.signal import butter, sosfilt
    n = int(secs * sr)
    if len(x) <= n:
        return 0
    hp = sosfilt(butter(4, 1000, "highpass", fs=sr, output="sos"), x)
    c = np.concatenate([[0.0], np.cumsum(hp.astype(np.float64) ** 2)])
    return int(np.argmax(c[n:] - c[:-n]))


def audio_variant(x, sr, kind):
    """Half speed reads the samples at sr/2, so a 10 kHz note lands at 5 kHz, which the model can hear."""
    rate = sr / 2 if kind.startswith("slow2") else sr
    n = float(kind.split("trim")[1])
    s = loudest(x, sr, n)
    return resample(x[s:s + int(n * sr)], rate, SR)


def load_audio(path):
    import librosa
    x, sr = librosa.load(path, sr=None, mono=True, duration=120)
    return np.asarray(x, np.float32), sr


# ---------------------------------------------------------------- prediction
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def blocks_norm(v, d=768):
    v = v.reshape(-1, d)
    return (v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)).reshape(-1)


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


def fuse(pi, pa):
    a, b = pi.max() ** GAMMA, pa.max() ** GAMMA
    return softmax((a * np.log(pi + 1e-9) + b * np.log(pa + 1e-9)) / (a + b))


def top(p, k=5):
    return {f"{COMMON[i]} ({SCI[i]})": float(p[i]) for i in np.argsort(-p)[:k]}


def identify(img, audio):
    if img is None and audio is None:
        raise gr.Error("Add a photo, a recording, or both.")
    pi = photo_probs(img) if img is not None else None
    pa = song_probs(audio) if audio is not None else None
    p = fuse(pi, pa) if pi is not None and pa is not None else (pi if pi is not None else pa)
    best = int(p.argmax())
    thumb = os.path.join(HERE, "thumbs", f"{CODES[best]}.jpg")
    return (top(p), top(pi) if pi is not None else None, top(pa) if pa is not None else None,
            thumb if os.path.exists(thumb) else None)


EX = os.path.join(HERE, "examples")
examples = [[os.path.join(EX, a) if a else None, os.path.join(EX, b) if b else None]
            for a, b in (("photo1.jpg", None), (None, "song1.mp3"), ("photo1.jpg", "song1.mp3"))
            if all(os.path.exists(os.path.join(EX, f)) for f in (a, b) if f)]

with gr.Blocks(title="burds: bird ID from photo + song") as demo:
    gr.Markdown(
        "# burds\nIdentify one of **60 North American birds** from a photo, a song, or both. "
        "One model ([EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)) embeds both, "
        "with no fine-tuning: small linear classifiers trained on openly licensed iNaturalist photos and "
        "xeno-canto recordings do the rest. [How it works and results](https://github.com/SreenityaThatikunta/burds)")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Photo", sources=["upload", "webcam", "clipboard"])
            aud = gr.Audio(type="filepath", label="Song or call", sources=["upload", "microphone"])
            btn = gr.Button("Identify", variant="primary")
        with gr.Column():
            out = gr.Label(num_top_classes=5, label="Best guess")
            ref = gr.Image(label="Reference photo (iNaturalist, see credits)", interactive=False, height=220)
            with gr.Accordion("Photo only / song only", open=False):
                out_i = gr.Label(num_top_classes=5, label="From the photo")
                out_a = gr.Label(num_top_classes=5, label="From the song")
    if examples:
        gr.Examples(examples, [img, aud])
    gr.Markdown("Works on CPU; a prediction takes a few seconds. Species list and media credits: "
                "`taxa.csv`, `credits.csv` in this Space.")
    btn.click(identify, [img, aud], [out, out_i, out_a, ref])

if __name__ == "__main__":
    demo.launch()
