# How many seconds of audio? (loudest N seconds, linear probe, audio test n=1264)

| seconds | normal speed | 95% CI | half speed | 95% CI |
|---|---|---|---|---|
| 1 | 24.1% | [21.8, 26.5] | 31.1% | [28.6, 33.5] |
| 2 | 27.8% | [25.4, 30.5] | 35.3% | [32.5, 37.8] |
| 3 | 27.5% | [25.0, 29.8] | 35.4% | [32.9, 38.0] |
| 5 | 27.8% | [25.3, 30.4] | 38.6% | [35.7, 41.3] |
| 7 | 27.0% | [24.5, 29.4] | 37.3% | [34.3, 40.0] |
| 10 (full clip) | 27.9% | [25.6, 30.4] | 37.4% | [34.9, 40.0] |

## Combining lengths

| features | dims | top-1 | 95% CI | top-5 |
|---|---|---|---|---|
| both speeds, all lengths | 9216 | 46.9% | [44.2, 49.4] | 74.1% |
| half speed, all lengths | 4608 | 44.6% | [42.1, 47.4] | 72.1% |
| step B: all five (previous best) | 3840 | 44.3% | [41.6, 47.2] | 71.9% |
| best normal (10s) + best half (5s) + full both speeds | 3072 | 41.5% | [38.8, 44.5] | 69.9% |
| normal speed, all lengths | 4608 | 33.9% | [31.2, 36.6] | 62.1% |

## Photo + song (paired test, n=1500, confidence fusion)

| audio features | image only | audio only | fused | gain vs image (95% CI) |
|---|---|---|---|---|
| step B: all five (previous best) | 93.5% | 43.6% | **96.3%** | [+1.5, +4.1] pts |
| both speeds, all lengths | 93.5% | 45.7% | **95.7%** | [+1.1, +3.6] pts |
