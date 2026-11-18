# Step B + C: audio variants and photo token budgets

Embedding wall time: **36.64 min** on 2 GPU(s).

## Photo token budget (40 photos per species subset; linear probe)

| tokens/photo | top-1 | 95% CI | centroid | photos/s per GPU | load wait | encode |
|---|---|---|---|---|---|---|
| 70 | 82.7% | [80.5, 85.0] | 78.3% | 10.72 | 0.8s | 223.0s |
| 140 | 87.9% | [86.1, 89.7] | 84.0% | 5.0 | 0.5s | 479.4s |
| 280 | 91.0% | [89.3, 92.5] | 86.0% | 2.13 | 0.4s | 1128.9s |

## Audio variants (full audio test set, n=1264; linear probe)

| variant | top-1 | 95% CI | top-5 | centroid |
|---|---|---|---|---|
| all five concatenated | 44.3% | [41.5, 46.9] | 71.9% | – |
| full + half speed + loudest 3 s | 42.2% | [39.5, 44.8] | 70.3% | – |
| full + half speed (concatenated) | 40.7% | [38.1, 43.4] | 69.0% | – |
| half speed | 37.4% | [34.8, 40.1] | 69.1% | 24.0% |
| loudest 3 s, half speed | 35.4% | [33.1, 38.1] | 65.7% | 21.1% |
| 3 s windows, probe per window, vote | 29.9% | [27.4, 32.3] | 57.8% | – |
| mean of 3 s windows | 28.6% | [26.2, 31.1] | 58.1% | 12.3% |
| full clip (run 2 setup) | 27.9% | [25.6, 30.2] | 55.9% | 15.8% |
| loudest 3 s | 27.5% | [25.0, 29.8] | 54.2% | 15.8% |

## Audio throughput

| job | clips/s per GPU | inputs | load wait | encode |
|---|---|---|---|---|
| aud_base | 8.88 | 3861 | 0.4s | 434.2s |
| aud_trim3 | 21.34 | 3861 | 0.6s | 180.3s |
| aud_slow2 | 4.41 | 3861 | 0.4s | 875.7s |
| aud_slow2_trim3 | 12.72 | 3861 | 0.6s | 302.9s |
| aud_win3 | 6.88 | 15444 | 0.5s | 560.7s |

## Photo + song fusion (paired test, n=1500)

| audio features | image only | audio only | weighted | confidence | gain vs image (95% CI) |
|---|---|---|---|---|---|
| full clip (run 2 setup) | 93.5% | 28.5% | 94.5% | 94.7% | [+0.3, +2.2] pts |
| all five concatenated | 93.5% | 43.6% | 96.2% | 96.3% | [+1.6, +4.0] pts |
