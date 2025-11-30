"""
Augment HarmonyAI dataset with new raw data.

This script:
- Reads all raw CSV files from data/raw/
- Cleans and normalizes new data
- Merges with existing processed dataset
- Deduplicates across old and new data
- Saves updated clean dataset
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import load_raw_csvs, clean_songs, save_clean

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def augment_dataset(
    raw_dir: str,
    processed_path: str,
    min_lyric_chars: int = 40
) -> None:
    """
    Augment existing dataset with new raw data.

    Args:
        raw_dir: Directory containing raw CSV files
        processed_path: Path to existing processed dataset
        min_lyric_chars: Minimum lyric length for filtering
    """
    logger.info("=" * 60)
    logger.info("Augmenting Dataset")
    logger.info("=" * 60)

    # Load and clean new raw data
    logger.info(f"Loading raw data from {raw_dir}")
    try:
        raw_df = load_raw_csvs(raw_dir)
        new_rows_in = len(raw_df)
        logger.info(f"Loaded {new_rows_in} raw rows")
    except ValueError as e:
        logger.error(f"Failed to load raw data: {e}")
        sys.exit(1)

    logger.info("Cleaning new data...")
    cleaned_new = clean_songs(raw_df, min_lyric_chars=min_lyric_chars)
    new_rows_clean = len(cleaned_new)
    logger.info(f"After cleaning: {new_rows_clean} rows")

    # Load existing processed data if it exists
    processed_path_obj = Path(processed_path)
    if processed_path_obj.exists():
        logger.info(f"Loading existing processed data from {processed_path}")
        existing_df = pd.read_csv(processed_path)
        existing_count = len(existing_df)
        logger.info(f"Existing dataset: {existing_count} rows")

        # Merge new and existing
        logger.info("Merging datasets...")
        combined = pd.concat([existing_df, cleaned_new], ignore_index=True)
        before_dedup = len(combined)

        # Deduplicate across entire combined dataset
        logger.info("Deduplicating merged dataset...")
        combined['_dedup_key'] = (
            combined['title'].str.lower() + '|||' + combined['artist'].str.lower()
        )
        combined = combined.drop_duplicates(subset='_dedup_key', keep='first')
        combined = combined.drop(columns=['_dedup_key'])
        after_dedup = len(combined)

        duplicates_removed = before_dedup - after_dedup
        new_songs_added = after_dedup - existing_count

        logger.info(f"Duplicates removed: {duplicates_removed}")
        logger.info(f"New songs added: {new_songs_added}")

    else:
        logger.info("No existing processed data found - using only new data")
        combined = cleaned_new
        existing_count = 0
        duplicates_removed = 0
        new_songs_added = len(combined)

    # Reset index
    combined = combined.reset_index(drop=True)

    # Save updated dataset
    logger.info(f"Saving augmented dataset to {processed_path}")
    save_clean(combined, processed_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("Augmentation Complete")
    logger.info("=" * 60)
    logger.info(f"Raw rows in:            {new_rows_in:,}")
    logger.info(f"Cleaned rows:           {new_rows_clean:,}")
    logger.info(f"Existing dataset:       {existing_count:,}")
    logger.info(f"Duplicates removed:     {duplicates_removed:,}")
    logger.info(f"New songs added:        {new_songs_added:,}")
    logger.info(f"Final dataset size:     {len(combined):,}")
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Augment HarmonyAI dataset with new raw data'
    )
    parser.add_argument(
        '--raw_dir',
        default='data/raw',
        help='Directory containing raw CSV files (default: data/raw)'
    )
    parser.add_argument(
        '--processed',
        default='data/processed/songs_clean.csv',
        help='Path to processed dataset (default: data/processed/songs_clean.csv)'
    )
    parser.add_argument(
        '--min_lyric_chars',
        type=int,
        default=40,
        help='Minimum lyric length (default: 40)'
    )

    args = parser.parse_args()

    try:
        augment_dataset(args.raw_dir, args.processed, args.min_lyric_chars)
        logger.info("SUCCESS")
        sys.exit(0)
    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
