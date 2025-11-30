"""
Build embeddings for HarmonyAI song dataset.

This script:
- Loads cleaned song dataset
- Generates embeddings using local HashingVectorizer
- Saves embeddings and metadata
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import embed_texts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_embeddings(
    input_path: str,
    output_dir: str,
    batch_size: int = 100
) -> None:
    """
    Build embeddings from cleaned song dataset.

    Args:
        input_path: Path to cleaned CSV file
        output_dir: Directory to save embeddings and metadata
        batch_size: Number of songs to process at once
    """
    logger.info("=" * 60)
    logger.info("Building Embeddings")
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

    # Generate embeddings
    logger.info(f"Generating embeddings (batch_size={batch_size})...")
    lyrics_list = df['lyrics'].fillna('').astype(str).tolist()

    all_embeddings = []
    num_batches = (len(lyrics_list) + batch_size - 1) // batch_size

    for i in range(0, len(lyrics_list), batch_size):
        batch_num = i // batch_size + 1
        batch = lyrics_list[i:i + batch_size]

        logger.info(f"Processing batch {batch_num}/{num_batches} ({len(batch)} songs)")
        batch_embeddings = embed_texts(batch)
        all_embeddings.append(batch_embeddings)

    # Concatenate all embeddings
    embeddings_matrix = np.vstack(all_embeddings)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    embeddings_file = output_path / 'song_embed.npy'
    logger.info(f"Saving embeddings to {embeddings_file}")
    np.save(embeddings_file, embeddings_matrix)

    # Save metadata
    metadata = df[['song_id', 'title', 'artist']].copy()
    metadata_path = output_path / 'songs_meta.parquet'
    logger.info(f"Saving metadata to {metadata_path}")
    metadata.to_parquet(metadata_path, index=False)

    # Print diagnostics
    logger.info("=" * 60)
    logger.info("Embedding Build Complete")
    logger.info("=" * 60)
    logger.info(f"Number of songs:        {embeddings_matrix.shape[0]:,}")
    logger.info(f"Embedding dimensions:   {embeddings_matrix.shape[1]:,}")
    logger.info(f"Matrix size:            {embeddings_matrix.nbytes / 1024 / 1024:.2f} MB")
    logger.info(f"Avg L2 norm:            {np.linalg.norm(embeddings_matrix, axis=1).mean():.4f}")
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Build embeddings for HarmonyAI'
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
        help='Output directory for embeddings (default: data/index)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size for processing (default: 100)'
    )

    args = parser.parse_args()

    try:
        build_embeddings(args.input_path, args.out_dir, args.batch_size)
        logger.info("SUCCESS")
        sys.exit(0)
    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
