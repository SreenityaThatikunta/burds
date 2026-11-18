# burds

Identify bird species from a **photo, a recording of its song, or both**, using
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), with no model training. Every
image and sound clip becomes a 768-d vector in one shared space, and a new observation is labeled by
its nearest species.

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
