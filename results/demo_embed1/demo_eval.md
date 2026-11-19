# Demo model: trained only on openly licensed web data

Training data: 1800 iNaturalist photos, 1195 xeno-canto recordings (60 species). 5-fold CV top-1: photo 82.6%, song 57.1%, photo + song 92.1% (gamma=0.5).

## Tested on SSW60 (different photographers, recordists and sites; never seen)

| input | n | top-1 | top-5 |
|---|---|---|---|
| photo | 5171 | 88.0% | 99.4% |
| song | 1264 | 31.4% | 56.8% |
| photo + song (paired) | 1500 | **88.5%** | |

Paired set: photo only 87.9%, gain from adding song [-0.9, +2.3] pts (95% CI).
