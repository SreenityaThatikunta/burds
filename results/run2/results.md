# SSW60 baseline — EmbeddingGemma 2, no training

60 species · images: NABirds split · audio: SSW60 audio_ml split · Tesla T4 · embedding time 127.8 min

## Paired test set (top-1, nearest class centroid)

| dims | image only | audio only | late fusion | joint embedding | joint + late |
|---|---|---|---|---|---|
| 768 | 88.7% | 17.2% | 88.3% | 87.0% | 89.1% |
| 512 | 88.3% | 17.1% | 88.3% | 86.7% | 89.3% |
| 256 | 87.9% | 16.8% | 87.5% | 85.9% | 88.5% |
| 128 | 86.7% | 16.3% | 86.7% | 84.5% | 87.7% |

## Full single-modality test sets

| dims | modality | centroid top-1 | centroid top-5 | 1-NN top-1 | n |
|---|---|---|---|---|---|
| 768 | image | 88.6% | 99.2% | 89.9% | 5171 |
| 768 | audio | 15.8% | 38.1% | 15.2% | 1264 |
| 512 | image | 88.1% | 99.1% | 90.1% | 5171 |
| 512 | audio | 16.0% | 38.1% | 15.5% | 1264 |
| 256 | image | 87.8% | 99.2% | 89.7% | 5171 |
| 256 | audio | 16.0% | 37.7% | 16.3% | 1264 |
| 128 | image | 87.3% | 99.1% | 88.5% | 5171 |
| 128 | audio | 15.5% | 37.5% | 14.6% | 1264 |

## Zero-shot (match against species names, no examples)

| dims | image→name | audio→name | pair→name |
|---|---|---|---|
| 768 | 68.3% | 4.4% | 64.4% |
| 512 | 67.4% | 5.1% | 64.2% |
| 256 | 65.8% | 4.3% | 61.3% |
| 128 | 53.9% | 2.9% | 53.7% |

## Hardest species (768-d, paired)

| species | image | audio | fused |
|---|---|---|---|
| Cape May Warbler | 80.0% | 4.0% | 24.0% |
| Common Raven | 40.0% | 16.0% | 56.0% |
| Downy Woodpecker | 76.0% | 16.0% | 64.0% |
| Rusty Blackbird | 84.0% | 8.0% | 68.0% |
| Cooper's Hawk | 64.0% | 0.0% | 72.0% |
| Purple Finch | 76.0% | 0.0% | 72.0% |
| Red-winged Blackbird | 76.0% | 12.0% | 72.0% |
| Sharp-shinned Hawk | 68.0% | 36.0% | 76.0% |
| Hairy Woodpecker | 56.0% | 68.0% | 76.0% |
| Chipping Sparrow | 80.0% | 24.0% | 76.0% |
