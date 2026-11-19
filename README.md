# burds

Identify bird species from a **photo, a recording of its song, or both**, using
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), with no model training. Every
image and sound clip becomes a 768-d vector in one shared space, and a new observation is labeled by
its nearest species.

![burds demo: photo only, song only, photo + song](docs/burds_demo.gif)

The demo app ([`app/`](app/), [MP4](docs/burds_demo.mp4)) on photos and recordings from iNaturalist
that the model never saw in training. **These examples were picked to work:** the photos were right
on the first or second try, but the song-only clip took 14 tries (Carolina Wren failed 8 times
first). Measured accuracy on unseen data is below: photo 88%, song 36%. The video was recorded before
the song training improved (below). Recorded headlessly on
Kaggle by [`kaggle/demo_video`](kaggle/demo_video/video_template.py); credits are in
[`results/demo_video2/media.json`](results/demo_video2/media.json).

## Layout
- `kaggle/ssw60_baseline/`: Kaggle kernel that streams [SSW60](https://github.com/visipedia/ssw60)
  (photos + audio only), embeds everything on a GPU, and reports accuracy for image-only, audio-only,
  fused, zero-shot, and truncated (768/512/256/128-d) embeddings.
- `results/`: metrics pulled back from Kaggle (`results.md`, `results.json`).

All data download and compute happens on Kaggle; nothing large runs locally.

## Baseline results (run 2, no training)
60 species from SSW60. Each test item is labeled by its nearest species centroid, built from the
training embeddings. Full tables: [`results/run2/results.md`](results/run2/results.md).

| | top-1 (768-d) | top-1 (256-d) |
|---|---|---|
| Photo only (NABirds test, n=5,171) | **88.6%** (top-5 99.2%) | 87.8% |
| Song only (SSW60 audio test, n=1,264) | 15.8% (top-5 38.1%) | 16.0% |
| Photo + song, equal-weight score sum (synthetic pairs) | 88.3% | 87.5% |
| Photo → species *name*, zero-shot | 68.3% | 65.8% |

Random guessing is 1.7%. The photos work very well out of the box. The audio embeddings are only
weakly specific to species, so giving them equal weight doesn't help fusion.

## Linear classifiers and smarter fusion (frozen embeddings, CPU only)
Logistic regression on top of the frozen embeddings; the model itself is not fine-tuned. Every
setting is chosen by 5-fold cross-validation on the train split. 95% bootstrap CIs. Full report:
[`results/analysis1/analysis.md`](results/analysis1/analysis.md).

| | nearest centroid | k-NN | linear classifier |
|---|---|---|---|
| Photo only | 88.6% | 91.9% | **94.4%** [93.7, 95.0] (top-5 99.7%) |
| Song only | 15.8% | 18.4% | **26.3%** [23.7, 28.7] |

On the paired test set (n=1,500), photo only gets 93.9%. Confidence-weighted photo + song fusion
gets **95.3%**, and a fixed 0.6/0.4 weighting gets 94.8%. At 256 dimensions the fused result is
still 94.0%.

**Where it fails:** most photo errors are between look-alike species: Hairy vs. Downy Woodpecker,
Sharp-shinned vs. Cooper's Hawk, and American Crow vs. Common Raven. The song fixes some of these
(Hairy Woodpecker goes from 64% to 84%, Cooper's Hawk from 68% to 80%), but a weak song guess can
still pull a correct photo guess off (Downy Woodpecker drops from 84% to 76%).
Confusion matrices: [photo](results/analysis1/confusion_image.png) ·
[song](results/analysis1/confusion_audio.png) · [fused](results/analysis1/confusion_fused.png).

## Making the model hear birds: the half-speed trick
EmbeddingGemma 2 takes audio at 16 kHz, so it can't hear anything above 8 kHz, and many songbirds
sing higher than that. Playing each clip at **half speed** moves 0–11 kHz content into the range
the model can hear. Linear classifier, full audio test set (n=1,264). Full report:
[`results/variants1/variants.md`](results/variants1/variants.md).

| Audio input | top-1 | 95% CI |
|---|---|---|
| Full 10 s clip (previous setup) | 27.9% | [25.6, 30.2] |
| Loudest 3 s only | 27.5% | [25.0, 29.8] |
| **Half speed** | **37.4%** | [34.8, 40.1] |
| Full + half speed + loudest 3 s + windows (concatenated) | **44.3%** | [41.5, 46.9] |

With the improved audio, photo + song reaches **96.3%** on the paired test set, compared with
93.5% for photo only. The gain is **+2.8 points, 95% CI [+1.6, +4.0]**, so it is clearly above zero.

### How many seconds of audio?
The 3 s window came from BirdNET's chunk size, so we swept the loudest 1–10 s (energy above
1 kHz) at both speeds. Linear classifier, audio test set (n=1,264). Full report:
[`results/audiolen1/audiolen.md`](results/audiolen1/audiolen.md).

| seconds | normal speed | half speed |
|---|---|---|
| 1 | 24.1% | 31.1% |
| 2 | 27.8% | 35.3% |
| 3 | 27.5% | 35.4% |
| **5** | 27.8% | **38.6%** |
| 7 | 27.0% | 37.3% |
| 10 (full clip) | 27.9% | 37.4% |

- At normal speed, length barely matters from 2 s on: the model can't hear the high notes anyway.
- At half speed, 1 s is too short and accuracy peaks around **5 s**. The 5 s clip is within the CI
  of the full clip, so it is the cheaper choice for a live demo rather than a clear win.
- Concatenating every length at both speeds (9,216 dims) gives the best audio-only result,
  **46.9%** [44.2, 49.4] (vs 44.3%), but no extra gain once fused with photos (95.7% vs 96.3%,
  overlapping CIs). Photo + song is saturated by the photo side, so the five-version setup stays.

## Demo: trained only on openly licensed web data
SSW60 can't be redistributed, so the demo ([`app/`](app/)) learns from **1,800 iNaturalist photos**
(CC0 / CC BY / CC BY-SA), **1,195 xeno-canto recordings** and **1,746 iNaturalist sound
recordings**, every file credited in [`app/credits.csv`](app/credits.csv). The model itself is
still untouched; only two linear classifiers are trained. Two species had to be looked up under
newer scientific names (Cooper's Hawk → *Astur cooperii*, Hairy Woodpecker → *Leuconotopicus villosus*).

Tested on the SSW60 test sets, which come from different photographers, recordists and sites
([`results/demo_embed3/demo_eval.md`](results/demo_embed3/demo_eval.md)):

| input | trained on web data | trained on SSW60 (above) |
|---|---|---|
| photo (n=5,171) | **88.0%** (top-5 99.4%) | 94.4% |
| song (n=1,264) | 36.1% (top-5 65.0%) | 44.3% |
| photo + song (paired, n=1,500) | **90.8%**, +2.9 pts over photo, 95% CI [+1.4, +4.4] | 96.3% |

Photos transfer well: about 6 points lost to a new domain. Songs transfer worse, and the training
recordings matter more than anything else. Three song recipes, chosen by cross-validation on the
web data (SSW60 only reported, never used to choose):

| song training data | rows | CV top-1 | SSW60 song top-1 | top-5 |
|---|---|---|---|---|
| xeno-canto only (clean, focal recordings) | 1,195 | 34.4% | 31.5% | 57.1% |
| **+ iNaturalist field and phone recordings** | 2,941 | **41.4%** | **36.1%** | **65.0%** |
| + iNaturalist + background-noise copies | 8,823 | 38.6% | 34.2% | 61.3% |

Adding real field recordings lifts song accuracy by 4.6 points and makes the photo + song gain
significant again. Adding *synthetic* background noise (the quietest stretch of another recording,
0-12 dB SNR) hurts: real noisy recordings beat simulated ones here.

## Speed
All seven embedding jobs (3 photo budgets on 2,400 photos, 5 audio versions of 3,861 clips) take
**37 min** on Kaggle's 2× T4, compared with 4.5 h for a single set in run 2. The speedup comes
from reusing the prepped data (no 33 GB download), one worker per GPU, and photo decoding that
never makes the GPU wait. Photo token budget is a direct speed/accuracy trade-off (linear
classifier, 20 training photos per species):

| tokens per photo | top-1 | photos/s per GPU |
|---|---|---|
| 70 | 82.7% | 10.7 |
| 140 | 87.9% | 5.0 |
| 280 (default) | 91.0% | 2.1 |

## Run
```bash
kaggle kernels push -p kaggle/ssw60_baseline
kaggle kernels status sreenityathatikunta/burds-ssw60-baseline
kaggle kernels output sreenityathatikunta/burds-ssw60-baseline -p results --file-pattern 'results\..*|taxa\.csv'
```

## Data and licensing
SSW60 media is for **research use only and must not be redistributed**. This repo never stores SSW60
images or audio, and the kernel is private. The public demo will use openly licensed
iNaturalist photos and xeno-canto recordings, with credit to each author.

## Citation
Van Horn et al., *Exploring Fine-grained Audiovisual Categorization with the SSW60 Dataset*, ECCV 2022.
