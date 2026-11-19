# Demo model: trained only on openly licensed web data

Training data: 1800 iNaturalist photos, 1195 xeno-canto + 1746 iNaturalist recordings (60 species); song recipe: xeno-canto + iNaturalist. 5-fold CV top-1: photo 82.6%, song 41.4%, photo + song 88.8% (gamma=1).

## Tested on SSW60 (different photographers, recordists and sites; never seen)

| input | n | top-1 | top-5 |
|---|---|---|---|
| photo | 5171 | 88.0% | 99.4% |
| song | 1264 | 36.1% | 65.0% |
| photo + song (paired) | 1500 | **90.8%** | |

Paired set: photo only 87.9%, gain from adding song [+1.4, +4.4] pts (95% CI).

## Song training recipes (chosen by CV on the web data; SSW60 shown for reference)

| recipe | training rows | CV, all clean recordings | CV, iNaturalist only | SSW60 song top-1 | top-5 |
|---|---|---|---|---|---|
| xeno-canto | 1195 | 34.4% | 19.3% | 31.5% | 57.1% |
| **xeno-canto + iNaturalist** | 2941 | 41.4% | 31.7% | 36.1% | 65.0% |
| xeno-canto + iNaturalist + noise | 8823 | 38.6% | 28.4% | 34.2% | 61.3% |
