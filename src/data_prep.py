"""
Data ingestion and cleaning for HarmonyAI music dataset.

This module handles:
- Loading raw CSV files from multiple sources
- Normalizing schema to standardized format
- Cleaning and deduplication
- Deterministic ID generation
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_csvs(raw_dir: str) -> pd.DataFrame:
    """
    Load all CSV files from raw directory and normalize to standard schema.

    Args:
        raw_dir: Path to directory containing raw CSV files

    Returns:
        DataFrame with columns: song_id, title, artist, lyrics

    Raises:
        ValueError: If no CSV files found or if required columns are missing
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise ValueError(f"Raw directory does not exist: {raw_dir}")

    csv_files = list(raw_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {raw_dir}")

    logger.info(f"Found {len(csv_files)} CSV file(s) in {raw_dir}")

    all_frames = []
    total_rows = 0

    for csv_file in csv_files:
        logger.info(f"Loading {csv_file.name}")

        try:
            df = pd.read_csv(csv_file)
            rows_loaded = len(df)
            total_rows += rows_loaded

            # Normalize column names (handle variations)
            df.columns = df.columns.str.lower().str.strip()

            # Ensure required columns exist
            required_cols = {'title', 'artist', 'lyrics'}
            if not required_cols.issubset(set(df.columns)):
                logger.warning(
                    f"Skipping {csv_file.name}: missing required columns. "
                    f"Has: {list(df.columns)}, needs: {list(required_cols)}"
                )
                continue

            # Generate deterministic song_id if not present
            if 'song_id' not in df.columns:
                df['song_id'] = df.apply(
                    lambda row: abs(hash((str(row['title']) + str(row['artist'])).lower())) % (10**9),
                    axis=1
                )

            # Select and reorder columns
            df = df[['song_id', 'title', 'artist', 'lyrics']].copy()

            all_frames.append(df)
            logger.info(f"Loaded {rows_loaded} rows from {csv_file.name}")

        except Exception as e:
            logger.error(f"Error loading {csv_file.name}: {e}")
            continue

    if not all_frames:
        raise ValueError("No valid CSV files could be loaded")

    # Concatenate all dataframes
    result = pd.concat(all_frames, ignore_index=True)
    logger.info(f"Total rows loaded: {total_rows}")
    logger.info(f"Combined dataset shape: {result.shape}")

    return result


def clean_songs(
    df: pd.DataFrame,
    min_lyric_chars: int = 40
) -> pd.DataFrame:
    """
    Clean and deduplicate song dataset.

    Operations performed:
    - Strip and lowercase lyrics
    - Remove blank/null lyrics
    - Filter out songs with lyrics below minimum length
    - Deduplicate by normalized (title, artist) pair

    Args:
        df: Input DataFrame with columns: song_id, title, artist, lyrics
        min_lyric_chars: Minimum number of characters required in lyrics

    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)
    logger.info(f"Starting cleaning with {initial_count} songs")

    # Create a working copy
    cleaned = df.copy()

    # Ensure all text columns are strings
    for col in ['title', 'artist', 'lyrics']:
        cleaned[col] = cleaned[col].astype(str)

    # Strip whitespace from all text columns
    cleaned['title'] = cleaned['title'].str.strip()
    cleaned['artist'] = cleaned['artist'].str.strip()
    cleaned['lyrics'] = cleaned['lyrics'].str.strip()

    # Remove rows with null or empty lyrics
    before_null_removal = len(cleaned)
    cleaned = cleaned[cleaned['lyrics'].notna()]
    cleaned = cleaned[cleaned['lyrics'].str.len() > 0]
    null_removed = before_null_removal - len(cleaned)
    if null_removed > 0:
        logger.info(f"Removed {null_removed} songs with null/empty lyrics")

    # Filter by minimum lyric length
    before_length_filter = len(cleaned)
    cleaned = cleaned[cleaned['lyrics'].str.len() >= min_lyric_chars]
    length_filtered = before_length_filter - len(cleaned)
    if length_filtered > 0:
        logger.info(f"Removed {length_filtered} songs with lyrics < {min_lyric_chars} chars")

    # Create normalized key for deduplication
    cleaned['_dedup_key'] = (
        cleaned['title'].str.lower() + '|||' + cleaned['artist'].str.lower()
    )

    # Deduplicate - keep first occurrence
    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset='_dedup_key', keep='first')
    duplicates_removed = before_dedup - len(cleaned)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate songs")

    # Remove temporary dedup column
    cleaned = cleaned.drop(columns=['_dedup_key'])

    # Reset index
    cleaned = cleaned.reset_index(drop=True)

    final_count = len(cleaned)
    total_removed = initial_count - final_count

    logger.info(f"Cleaning complete: {final_count} songs remaining")
    logger.info(f"Total removed: {total_removed} ({100 * total_removed / initial_count:.1f}%)")

    # Log summary statistics
    logger.info(f"Unique artists: {cleaned['artist'].nunique()}")
    logger.info(f"Avg lyric length: {cleaned['lyrics'].str.len().mean():.0f} chars")

    return cleaned


def save_clean(df: pd.DataFrame, path: str) -> None:
    """
    Save cleaned dataset to CSV.

    Args:
        df: Cleaned DataFrame
        path: Output file path
    """
    output_path = Path(path)

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(df)} songs to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
