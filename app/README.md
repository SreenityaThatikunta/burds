---
title: burds
emoji: 🐦
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Identify 60 North American birds from a photo and/or song
---

# burds

Identify one of 60 North American birds from a **photo, a song, or both**, with one multimodal
embedding model ([EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)) and no fine-tuning.

- Photos are embedded at 280 image tokens; songs are embedded three ways (half speed loudest 5 s,
  half speed loudest 10 s, normal speed loudest 10 s). Half speed moves high-pitched song into the
  0-8 kHz range the model hears.
- Two small linear classifiers, trained only on openly licensed iNaturalist photos and xeno-canto
  recordings, read the embeddings; photo and song are combined by confidence.

Code, experiments and results: https://github.com/SreenityaThatikunta/burds

## Credits
Reference media: iNaturalist observers (CC0 / CC BY / CC BY-SA) and xeno-canto recordists (Creative
Commons). Every file, author and licence is listed in `credits.csv`. Only embeddings and one
thumbnail per species ship with this Space.
