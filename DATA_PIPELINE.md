# HarmonyAI Data Pipeline (Short Version)

Goal: show how raw lyrics become indexed data for the recommender.

## One-Glance Map
```
data/raw/*.csv  -> augment_data  -> data/processed/songs_clean.csv
                          |-> build_index -> data/index/{tfidf_vectorizer.joblib,song_tfidf.npz,songs_meta.parquet}
                          |-> build_embeddings (optional) -> data/index/song_embed.npy
augment_data/index/embed -> eval -> reports/metrics.csv
```

## Commands to Run
Install deps once:
```bash
pip install -r requirements.txt
```

1) Clean + merge raw CSVs:
```bash
python -m scripts.augment_data
```
Output: `data/processed/songs_clean.csv` (validated, deduped, lyrics >=40 chars).

2) Build TF-IDF index:
```bash
python -m scripts.build_index
```
Output: `data/index/tfidf_vectorizer.joblib`, `data/index/song_tfidf.npz`, `data/index/songs_meta.parquet`.

3) Optional embeddings:
```bash
python -m scripts.build_embeddings
```
Output: `data/index/song_embed.npy`.

4) Evaluate:
```bash
python -m scripts.eval
```
Output: `reports/metrics.csv` (Precision@K, nDCG@K).

## Inputs/Expectations
- Raw CSV schema: `title, artist, lyrics` (case-insensitive). `song_id` optional.
- Files go in `data/raw/`. Missing columns or empty lyrics will fail validation.

## Quick Troubleshooting
- No CSVs found: put files in `data/raw/` with `.csv` extension.
- Missing columns error: ensure the three required headers exist.
- Low metrics: check queries in `tests/test_queries.json` match artists in your data; rerun `build_index` after adding data.

## What to show a teacher
- Clean dataset: `data/processed/songs_clean.csv`
- Indices: `data/index/*`
- Eval results: `reports/metrics.csv`
