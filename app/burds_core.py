"""
Pure numpy/scipy core of burds: audio preprocessing and score fusion. No model, no UI, so it is
fast to test (tests/) and shared by the app and the Kaggle kernels.

The audio functions must stay byte-identical to the copies in kaggle/demo_embed/demo_embed.py
(Kaggle kernels are single files); tests/test_consistency.py checks this.
"""
import numpy as np

SR = 16000  # EmbeddingGemma 2 audio rate: it hears 0-8 kHz


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


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)


def add_background(x, sr, noise, sr_n, snr_db):
    """Mix in the quietest stretch of another recording (wind, traffic, other birds, hiss) at snr_db."""
    if sr_n != sr:
        noise = resample(noise, sr_n, sr)
    n = min(len(x), len(noise))
    c = np.concatenate([[0.0], np.cumsum(noise.astype(np.float64) ** 2)])
    s0 = int(np.argmin(c[n:] - c[:-n])) if len(noise) > n else 0
    seg = np.resize(noise[s0:s0 + n], len(x))  # tile if the background is shorter than the clip
    return (x + seg * (rms(x) / rms(seg)) * 10 ** (-snr_db / 20)).astype(np.float32)


def audio_variant(x, sr, kind):
    """kind: [slow2_]trim<N>. Half speed reads the samples at sr/2, so frequency f lands at f/2."""
    rate = sr / 2 if kind.startswith("slow2") else sr
    n = float(kind.split("trim")[1])
    s = loudest(x, sr, n)
    return resample(x[s:s + int(n * sr)], rate, SR)


# ---------------------------------------------------------------- scores
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def blocks_norm(v, d=768):
    """L2-normalise each 768-d block of a concatenated embedding (one block per audio version)."""
    v = v.reshape(-1, d)
    return (v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)).reshape(-1)


def fuse(pi, pa, gamma):
    """Confidence fusion: average photo and song log-probs, each weighted by its top probability ** gamma."""
    a, b = pi.max() ** gamma, pa.max() ** gamma
    return softmax((a * np.log(pi + 1e-9) + b * np.log(pa + 1e-9)) / (a + b))


def top_k(p, names, k=5):
    return {names[i]: float(p[i]) for i in np.argsort(-p)[:k]}
