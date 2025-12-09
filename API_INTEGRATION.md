# HarmonyAI Music API Quick Guide

Purpose: pull ~500 real songs (metadata + lyrics) into the app using free APIs.

## Setup (5–7 minutes)
1) Install deps: `pip install -r requirements.txt`
2) Get Genius token: https://genius.com/api-clients → New API Client → Generate Access Token
3) Configure env:
```bash
cp .env.example .env
echo 'GENIUS_API_TOKEN=your_token_here' >> .env
```

## Fetch Songs (30–60 minutes runtime)
```bash
mkdir -p logs
python -m scripts.fetch_songs \
  --queries "rock" "pop" "indie" "alternative" "electronic" \
  --limit-per-query 100
```
What it does: searches MusicBrainz, grabs lyrics from Genius with lyrics.ovh fallback, writes to `data/songs.db`, exports `data/raw/songs_fetched.csv`.

## Plug Into Pipeline (2–3 minutes)
```bash
python -m scripts.augment_data      # clean + dedupe into data/processed/songs_clean.csv
python -m scripts.build_index       # TF-IDF index → data/index/
python -m scripts.build_embeddings  # optional hashing embeddings → data/index/
python -m scripts.eval              # metrics → reports/metrics.csv
```

## Optional: Streamlit on real data
Swap the demo loader with your clean catalog in `streamlit_app.py`:
```python
# use the processed dataset instead of the hardcoded demo
df = pd.read_csv('data/processed/songs_clean.csv')
```
Then run: `streamlit run streamlit_app.py`

## Minimal Troubleshooting
- Module missing: `pip install -r requirements.txt`
- Missing token: ensure `.env` has `GENIUS_API_TOKEN=...`
- Low lyric hit rate: try specific queries (artist names/decades), rerun
- DB locked: stop other fetch runs (`pkill -f fetch_songs`) or change `--db data/temp.db`

## Outputs to show a teacher
- `data/raw/songs_fetched.csv`: raw fetched songs
- `data/processed/songs_clean.csv`: cleaned catalog
- `data/index/*`: TF-IDF (and optional embeddings)
- `reports/metrics.csv`: evaluation scores
