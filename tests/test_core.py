"""Unit tests for app/burds_core.py: audio preprocessing and score fusion (no model needed)."""
import numpy as np
import pytest

from burds_core import SR, add_background, audio_variant, blocks_norm, fuse, loudest, resample, rms, softmax, top_k


def tone(freq, secs, sr):
    return np.sin(2 * np.pi * freq * np.arange(int(secs * sr)) / sr).astype(np.float32)


def peak_hz(x, sr):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[spec.argmax()]


def test_resample_length_and_dtype():
    y = resample(tone(1000, 2, 44100), 44100, SR)
    assert y.dtype == np.float32
    assert abs(len(y) - 2 * SR) <= 1


def test_half_speed_halves_frequency():
    # A 10 kHz note is above the model's 8 kHz limit; half speed moves it to 5 kHz.
    x = tone(10_000, 3, 44100)
    assert peak_hz(audio_variant(x, 44100, "slow2_trim2"), SR) == pytest.approx(5000, abs=20)
    assert peak_hz(audio_variant(tone(3000, 3, 44100), 44100, "trim2"), SR) == pytest.approx(3000, abs=20)


@pytest.mark.parametrize("kind,secs", [("trim5", 5), ("trim10", 10), ("slow2_trim5", 10), ("slow2_trim10", 20)])
def test_variant_output_length(kind, secs):
    # N s of input; half speed plays it at half rate, so it lasts 2N s at 16 kHz.
    y = audio_variant(np.random.default_rng(0).normal(size=44100 * 30).astype(np.float32), 44100, kind)
    assert abs(len(y) - secs * SR) <= 2


def test_short_clip_is_kept_whole():
    x = tone(3000, 2, 22050)
    assert loudest(x, 22050, 5) == 0
    assert abs(len(audio_variant(x, 22050, "trim5")) - 2 * SR) <= 2


def test_loudest_finds_the_bird_not_the_rumble():
    sr = 22050
    x = 0.01 * np.random.default_rng(0).normal(size=sr * 20).astype(np.float32)
    x += 2.0 * tone(150, 20, sr)  # loud low-frequency rumble everywhere (wind, traffic)
    x[12 * sr:15 * sr] += tone(4000, 3, sr)  # bird song at 12-15 s
    start = loudest(x, sr, 3) / sr
    assert 11.5 <= start <= 12.5


@pytest.mark.parametrize("snr", [0, 6, 12])
def test_add_background_hits_requested_snr(snr):
    x = tone(3000, 4, 44100)
    noise = np.random.default_rng(1).normal(size=22050 * 9).astype(np.float32)
    mixed = add_background(x, 44100, noise, 22050, snr)
    assert mixed.shape == x.shape
    assert 20 * np.log10(rms(x) / rms(mixed - x)) == pytest.approx(snr, abs=0.5)


def test_add_background_tiles_short_noise():
    x = tone(3000, 5, SR)
    mixed = add_background(x, SR, np.ones(SR, np.float32), SR, 10)
    assert mixed.shape == x.shape and np.isfinite(mixed).all()


def test_softmax_and_blocks_norm():
    p = softmax(np.array([1000.0, 1001.0, 999.0]))  # large logits must not overflow
    assert p.sum() == pytest.approx(1) and p.argmax() == 1
    v = blocks_norm(np.random.default_rng(0).normal(size=3 * 768))
    assert np.linalg.norm(v.reshape(3, 768), axis=1) == pytest.approx(np.ones(3))


def test_fuse_trusts_the_confident_modality():
    photo = np.array([0.90, 0.05, 0.05])  # confident: species 0
    song = np.array([0.30, 0.40, 0.30])  # unsure, leans to species 1
    p = fuse(photo, song, gamma=1.0)
    assert p.sum() == pytest.approx(1) and p.argmax() == 0
    assert fuse(song, photo, 1.0) == pytest.approx(p)  # symmetric in its inputs


def test_fuse_agreeing_inputs_stay_put():
    p = np.array([0.6, 0.3, 0.1])
    assert fuse(p, p, gamma=2.0) == pytest.approx(p, abs=1e-6)


def test_top_k_orders_by_probability():
    out = top_k(np.array([0.1, 0.7, 0.2]), ["a", "b", "c"], k=2)
    assert list(out) == ["b", "c"]
