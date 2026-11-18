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
weakly specific to species, so giving them equal weight doesn't help fusion yet. Next steps:
weighted or confidence-based fusion, and a linear classifier on top of the frozen embeddings.

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
