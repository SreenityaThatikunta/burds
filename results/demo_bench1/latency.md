# burds latency (end to end per request, median / p90, first call excluded)

Song clips: median 18 s long. CPU = 4 vCPUs (Kaggle).

| request | T4 GPU | CPU only |
|---|---|---|
| photo | 0.41 s / 0.41 s | 7.23 s / 7.32 s |
| song | 0.62 s / 0.67 s | 8.99 s / 9.17 s |
| photo + song | 1.04 s / 1.09 s | 16.29 s / 16.97 s |
