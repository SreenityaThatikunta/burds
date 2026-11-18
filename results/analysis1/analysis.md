# SSW60 analysis on frozen EmbeddingGemma 2 embeddings

All hyperparameters chosen by 5-fold CV on the train split; 95% bootstrap CIs on test.

## Single modality (full test sets)

| modality | method | top-1 | 95% CI | top-5 | n |
|---|---|---|---|---|---|
| image | centroid | 88.6% | [87.6, 89.4] | 99.2% | 5171 |
| image | knn | 91.9% | [91.2, 92.7] | 99.2% | 5171 |
| image | linear_probe | 94.4% | [93.7, 95.0] | 99.7% | 5171 |
| audio | centroid | 15.8% | [13.9, 17.8] | 38.1% | 1264 |
| audio | knn | 18.4% | [16.3, 20.6] | 42.4% | 1264 |
| audio | linear_probe | 26.3% | [23.7, 28.7] | 53.7% | 1264 |

## Photo + song (paired test set, n=1500)

| per-modality scorer | image only | audio only | equal weight | weighted | confidence |
|---|---|---|---|---|---|
| centroid | 88.7% | 17.2% | 88.3% | 89.7% | 90.0% |
| knn | 90.8% | 19.1% | 87.3% | 90.9% | 91.1% |
| linear_probe | 93.9% | 26.7% | 94.0% | 94.8% | 95.3% |

- Best fused (linear probe + weighted, image weight 0.60): **94.8%** [93.6, 95.9]
- Image-only on same pairs: 93.9% [92.7, 95.1]
- Joint single-embedding probe: 91.3%
- Early fusion probe on [image, audio]: 92.4%

## Embedding size (linear probe + weighted fusion, paired test)

| dims | image | audio | fused |
|---|---|---|---|
| 768 | 93.9% | 26.7% | 94.8% |
| 256 | 93.3% | 26.1% | 94.0% |
| 128 | 91.9% | 23.3% | 92.7% |

## Most confused (image)

| true | predicted as | rate |
|---|---|---|
| Hairy Woodpecker | Downy Woodpecker | 36.7% (22) |
| Sharp-shinned Hawk | Cooper's Hawk | 25.9% (30) |
| Cooper's Hawk | Sharp-shinned Hawk | 22.2% (26) |
| American Crow | Common Raven | 19.7% (15) |
| Common Raven | American Crow | 19.3% (11) |
| Downy Woodpecker | Hairy Woodpecker | 13.3% (8) |
| Cape May Warbler | Yellow-rumped Warbler | 11.7% (7) |
| Purple Finch | House Finch | 7.2% (7) |

## Most confused (audio)

| true | predicted as | rate |
|---|---|---|
| Scarlet Tanager | Red-eyed Vireo | 36.8% (7) |
| Common Raven | American Crow | 25.0% (5) |
| Barn Swallow | House Wren | 23.5% (4) |
| Red-bellied Woodpecker | Northern Flicker | 23.5% (4) |
| Cape May Warbler | Red-eyed Vireo | 22.2% (4) |
| Cooper's Hawk | Gray Catbird | 22.2% (4) |
| Wood Duck | Canada Goose | 21.1% (4) |
| Cedar Waxwing | Chipping Sparrow | 19.2% (5) |

## Most confused (fused)

| true | predicted as | rate |
|---|---|---|
| Common Raven | American Crow | 28.0% (7) |
| Downy Woodpecker | Hairy Woodpecker | 24.0% (6) |
| Hairy Woodpecker | Downy Woodpecker | 16.0% (4) |
| Sharp-shinned Hawk | Cooper's Hawk | 16.0% (4) |
| Northern Parula | Yellow-rumped Warbler | 12.0% (3) |
| Rusty Blackbird | Common Grackle | 12.0% (3) |
| Purple Finch | House Finch | 12.0% (3) |
| American Crow | Common Raven | 12.0% (3) |

## Where song helps / hurts most (paired, linear probe)

| species | image | audio | fused |
|---|---|---|---|
| Downy Woodpecker | 84.0% | 8.0% | 76.0% |
| Rusty Blackbird | 88.0% | 8.0% | 80.0% |
| Hooded Merganser | 100.0% | 40.0% | 96.0% |
| American Kestrel | 100.0% | 20.0% | 96.0% |
| Common Raven | 76.0% | 40.0% | 72.0% |
| Purple Finch | 88.0% | 0.0% | 84.0% |
| Eastern Phoebe | 80.0% | 20.0% | 88.0% |
| American Crow | 80.0% | 76.0% | 88.0% |
| Yellow-rumped Warbler | 92.0% | 4.0% | 100.0% |
| Sharp-shinned Hawk | 72.0% | 32.0% | 80.0% |
| Cooper's Hawk | 68.0% | 4.0% | 80.0% |
| Hairy Woodpecker | 64.0% | 76.0% | 84.0% |

Hyperparameters: `{"image": {"centroid_temp": 0.01, "knn_k": 10, "lr_C": 30}, "audio": {"centroid_temp": 0.01, "knn_k": 40, "lr_C": 30}, "fusion_centroid": {"image_weight": 0.9, "conf_gamma": 1.0}, "fusion_knn": {"image_weight": 0.9500000000000001, "conf_gamma": 2.0}, "fusion_linear_probe": {"image_weight": 0.6000000000000001, "conf_gamma": 1.0}}`
