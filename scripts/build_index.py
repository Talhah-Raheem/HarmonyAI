"""
Build TF-IDF index for HarmonyAI song dataset.

This script:
- Loads cleaned song dataset
- Fits TF-IDF vectorizer on song lyrics
- Saves vectorizer and TF-IDF matrix
- Saves song metadata for lookup
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_tfidf_index(
    input_path: str,
    output_dir: str
) -> None:
    """
    Build TF-IDF index from cleaned song dataset.

    Args:
        input_path: Path to cleaned CSV file
        output_dir: Directory to save index artifacts
    """
    logger.info("=" * 60)
    logger.info("Building TF-IDF Index")
    logger.info("=" * 60)

    # Load dataset
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} songs")

    # Validate required columns
    required_cols = {'song_id', 'title', 'artist', 'lyrics'}
    if not required_cols.issubset(set(df.columns)):
        logger.error(f"Missing required columns. Found: {list(df.columns)}")
        sys.exit(1)

    # Initialize TF-IDF vectorizer
    logger.info("Initializing TF-IDF vectorizer")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=1,
        sublinear_tf=True,
        strip_accents='unicode',
        lowercase=True
    )

    # Fit and transform
    logger.info("Fitting TF-IDF on song lyrics...")
    lyrics = df['lyrics'].fillna('').astype(str)
    tfidf_matrix = vectorizer.fit_transform(lyrics)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save TF-IDF vectorizer
    vectorizer_path = output_path / 'tfidf_vectorizer.joblib'
    logger.info(f"Saving vectorizer to {vectorizer_path}")
    joblib.dump(vectorizer, vectorizer_path)

    # Save TF-IDF matrix
    matrix_path = output_path / 'song_tfidf.npz'
    logger.info(f"Saving TF-IDF matrix to {matrix_path}")
    save_npz(matrix_path, tfidf_matrix)

    # Save metadata
    metadata = df[['song_id', 'title', 'artist']].copy()
    metadata_path = output_path / 'songs_meta.parquet'
    logger.info(f"Saving metadata to {metadata_path}")
    metadata.to_parquet(metadata_path, index=False)

    # Print diagnostics
    logger.info("=" * 60)
    logger.info("Index Build Complete")
    logger.info("=" * 60)
    logger.info(f"Vocabulary size:        {len(vectorizer.vocabulary_):,}")
    logger.info(f"Number of songs:        {tfidf_matrix.shape[0]:,}")
    logger.info(f"Feature dimensions:     {tfidf_matrix.shape[1]:,}")
    logger.info(f"Avg non-zero per doc:   {tfidf_matrix.nnz / tfidf_matrix.shape[0]:.1f}")
    logger.info(f"Sparsity:               {100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2f}%")
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Build TF-IDF index for HarmonyAI'
    )
    parser.add_argument(
        '--in',
        dest='input_path',
        default='data/processed/songs_clean.csv',
        help='Path to cleaned CSV file (default: data/processed/songs_clean.csv)'
    )
    parser.add_argument(
        '--out_dir',
        default='data/index',
        help='Output directory for index artifacts (default: data/index)'
    )

    args = parser.parse_args()

    try:
        build_tfidf_index(args.input_path, args.out_dir)
        logger.info("SUCCESS")
        sys.exit(0)
    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
