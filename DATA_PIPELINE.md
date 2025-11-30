# HarmonyAI Data Pipeline Documentation

## Overview

This document describes the complete data processing pipeline for HarmonyAI, including dataset creation, ingestion, cleaning, indexing, and evaluation.

## Directory Structure

```
harmonyai/
├── src/
│   ├── data_prep.py       # Data ingestion and cleaning
│   ├── validate.py        # Schema validation and statistics
│   └── embeddings.py      # Local embedding generation
├── scripts/
│   ├── augment_data.py    # Augment dataset with new data
│   ├── build_index.py     # Build TF-IDF index
│   ├── build_embeddings.py # Build embeddings index
│   └── eval.py            # Evaluate recommendation system
├── data/
│   ├── raw/               # Raw CSV files
│   ├── processed/         # Cleaned datasets
│   └── index/             # Index artifacts
├── tests/
│   ├── test_queries.json  # Test queries for evaluation
│   ├── test_data_prep.py  # Unit tests for data prep
│   └── test_validate.py   # Unit tests for validation
└── reports/               # Evaluation reports
```

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

## Pipeline Workflow

### Step 1: Prepare Raw Data

Place your raw CSV files in `data/raw/`. Each CSV should have the following columns:
- `title` (string): Song title
- `artist` (string): Artist name
- `lyrics` (string): Full song lyrics

Optional:
- `song_id` (integer): Unique identifier (auto-generated if missing)

Example CSV format:
```csv
title,artist,lyrics
Sunshine Days,Ava Lynn,Walking down the street with a smile on my face...
```

### Step 2: Clean and Process Data

Run the augment script to load, clean, and deduplicate the data:

```bash
python -m scripts.augment_data
```

This will:
- Load all CSV files from `data/raw/`
- Normalize schema to standard format
- Remove songs with lyrics shorter than 40 characters
- Remove duplicate songs (case-insensitive by title + artist)
- Remove songs with empty/null lyrics
- Merge with existing processed data (if any)
- Save to `data/processed/songs_clean.csv`

**Output:**
- `data/processed/songs_clean.csv` - Cleaned dataset

### Step 3: Build TF-IDF Index

Build the TF-IDF vectorizer and index:

```bash
python -m scripts.build_index
```

This creates:
- TF-IDF vectorizer with (1,2)-grams and 10,000 max features
- Sparse TF-IDF matrix for all songs
- Song metadata for lookup

**Outputs:**
- `data/index/tfidf_vectorizer.joblib` - Fitted TF-IDF vectorizer
- `data/index/song_tfidf.npz` - Sparse TF-IDF matrix
- `data/index/songs_meta.parquet` - Song metadata (ID, title, artist)

**Diagnostics printed:**
- Vocabulary size
- Number of songs
- Average non-zero entries per document
- Sparsity percentage

### Step 4: Build Embeddings (Optional)

Generate local embeddings using HashingVectorizer:

```bash
python -m scripts.build_embeddings
```

This creates:
- L2-normalized 512-dimensional embeddings for all song lyrics
- Same metadata as TF-IDF index

**Outputs:**
- `data/index/song_embed.npy` - Dense embedding matrix (N x 512)
- `data/index/songs_meta.parquet` - Song metadata

**Options:**
```bash
python -m scripts.build_embeddings --batch_size 50
```

### Step 5: Run Evaluation

Evaluate the recommendation system using test queries:

```bash
python -m scripts.eval
```

This will:
- Load test queries from `tests/test_queries.json`
- Use TF-IDF index to rank songs for each query
- Compute metrics: Precision@5, Precision@10, nDCG@5, nDCG@10
- Save results to `reports/metrics.csv`

**Output:**
- `reports/metrics.csv` - Evaluation metrics for all queries plus averages

**Sample output:**
```
                        query         relevant_artists  precision_at_5  precision_at_10  ndcg_at_5  ndcg_at_10
           happy and upbeat  Ava Lynn, Neon Wolves            0.400            0.200      0.500       0.350
  calm and peaceful night  Calm Harbor, City Nights          0.400            0.300      0.630       0.520
        nostalgic memories  Memory Lane, Retro Echo           0.200            0.200      0.387       0.280
      angry and aggressive  Red Static, Iron Path             0.600            0.400      0.774       0.650
                   AVERAGE                                    0.400            0.275      0.573       0.450
```

## Python API Usage

### Data Preparation

```python
from src.data_prep import load_raw_csvs, clean_songs, save_clean

# Load raw data
df = load_raw_csvs('data/raw')

# Clean data
cleaned = clean_songs(df, min_lyric_chars=40)

# Save cleaned data
save_clean(cleaned, 'data/processed/songs_clean.csv')
```

### Validation

```python
from src.validate import validate_schema, basic_stats, print_stats
import pandas as pd

# Load dataset
df = pd.read_csv('data/processed/songs_clean.csv')

# Validate schema
validate_schema(df)  # Raises ValueError if invalid

# Get statistics
stats = basic_stats(df)
print_stats(stats)
```

### Embeddings

```python
from src.embeddings import embed_texts, embed_single

# Generate embeddings for multiple texts
texts = ["happy song lyrics", "sad song lyrics"]
embeddings = embed_texts(texts)  # Returns (2, 512) array

# Generate embedding for single text
query = "upbeat and energetic"
query_embedding = embed_single(query)  # Returns (512,) array
```

## Script Options

### augment_data.py

```bash
python -m scripts.augment_data \
  --raw_dir data/raw \
  --processed data/processed/songs_clean.csv \
  --min_lyric_chars 40
```

### build_index.py

```bash
python -m scripts.build_index \
  --in data/processed/songs_clean.csv \
  --out_dir data/index
```

### build_embeddings.py

```bash
python -m scripts.build_embeddings \
  --in data/processed/songs_clean.csv \
  --out_dir data/index \
  --batch_size 100
```

### eval.py

```bash
python -m scripts.eval \
  --queries tests/test_queries.json \
  --dataset data/processed/songs_clean.csv \
  --index_dir data/index \
  --reports_dir reports
```

## Running Tests

Run all unit tests:

```bash
pytest tests/ -v
```

Run specific test files:

```bash
pytest tests/test_data_prep.py -v
pytest tests/test_validate.py -v
```

## Data Quality Checks

The pipeline includes multiple quality checks:

1. **Schema validation**: Ensures required columns exist
2. **Lyric length filtering**: Removes songs with insufficient lyrics (default: 40 chars)
3. **Deduplication**: Removes duplicate songs (case-insensitive)
4. **Null removal**: Removes songs with empty/null lyrics
5. **Type validation**: Ensures correct data types for all columns

## Metrics Explanation

- **Precision@K**: Fraction of top-K results that are relevant
- **nDCG@K**: Normalized Discounted Cumulative Gain - measures ranking quality with position weighting

Higher values indicate better performance (range: 0.0 to 1.0).

## Troubleshooting

### No CSV files found
- Ensure CSV files are in `data/raw/` directory
- Check file extensions are `.csv`

### Missing columns error
- Verify CSV has columns: `title`, `artist`, `lyrics`
- Column names are case-insensitive

### Low evaluation metrics
- Ensure test queries match song content
- Check that relevant artists exist in dataset
- Consider tuning TF-IDF parameters in `build_index.py`

## Integration with Streamlit UI

The processed data and indices can be loaded directly in the Streamlit app:

```python
import pandas as pd
import joblib
from scipy.sparse import load_npz

# Load cleaned dataset
df = pd.read_csv('data/processed/songs_clean.csv')

# Load TF-IDF index
vectorizer = joblib.load('data/index/tfidf_vectorizer.joblib')
tfidf_matrix = load_npz('data/index/song_tfidf.npz')

# Use for search and recommendations
query = "happy and upbeat"
query_vec = vectorizer.transform([query])
similarities = cosine_similarity(query_vec, tfidf_matrix)
```

## Next Steps

1. Add more songs to `data/raw/`
2. Run `augment_data.py` to update processed dataset
3. Rebuild indices with `build_index.py` and `build_embeddings.py`
4. Re-evaluate with `eval.py`
5. Monitor metrics in `reports/metrics.csv`

## Notes

- All scripts support `--help` flag for detailed options
- All scripts exit with non-zero status on errors
- Comprehensive logging included in all modules
- Fully offline operation (no external APIs required)
