"""
Schema validation and dataset statistics for HarmonyAI.

This module provides:
- Schema validation to ensure data integrity
- Statistical analysis of dataset quality
"""

from typing import Dict, Any

import pandas as pd
import numpy as np


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame contains required columns and proper types.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If schema validation fails
    """
    required_columns = {'song_id', 'title', 'artist', 'lyrics'}
    actual_columns = set(df.columns)

    missing_columns = required_columns - actual_columns

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}. "
            f"Found: {sorted(actual_columns)}"
        )

    # Check for empty dataframe
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    # Validate data types
    if not pd.api.types.is_integer_dtype(df['song_id']):
        raise ValueError("song_id must be integer type")

    for col in ['title', 'artist', 'lyrics']:
        if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
            raise ValueError(f"{col} must be string type")


def basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic statistics for the dataset.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary containing:
        - row_count: Total number of songs
        - unique_artists: Number of unique artists
        - lyric_length_min: Minimum lyric length
        - lyric_length_mean: Mean lyric length
        - lyric_length_median: Median lyric length
        - lyric_length_max: Maximum lyric length
        - null_counts: Dictionary of null counts per column
    """
    # Ensure lyrics column is string type for length calculation
    lyrics_series = df['lyrics'].astype(str)
    lyric_lengths = lyrics_series.str.len()

    stats = {
        'row_count': len(df),
        'unique_artists': df['artist'].nunique(),
        'unique_songs': df['song_id'].nunique(),
        'lyric_length_min': int(lyric_lengths.min()),
        'lyric_length_mean': float(lyric_lengths.mean()),
        'lyric_length_median': float(lyric_lengths.median()),
        'lyric_length_max': int(lyric_lengths.max()),
        'null_counts': df.isnull().sum().to_dict()
    }

    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    """
    Pretty-print dataset statistics.

    Args:
        stats: Statistics dictionary from basic_stats()
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total songs:        {stats['row_count']:,}")
    print(f"Unique artists:     {stats['unique_artists']:,}")
    print(f"Unique song IDs:    {stats['unique_songs']:,}")
    print()
    print("Lyric Length Statistics:")
    print(f"  Min:              {stats['lyric_length_min']:,} chars")
    print(f"  Mean:             {stats['lyric_length_mean']:,.0f} chars")
    print(f"  Median:           {stats['lyric_length_median']:,.0f} chars")
    print(f"  Max:              {stats['lyric_length_max']:,} chars")
    print()

    null_counts = stats['null_counts']
    total_nulls = sum(null_counts.values())

    if total_nulls > 0:
        print("Null Counts:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {col:20s}: {count:,}")
    else:
        print("Null Counts:        None")

    print("=" * 60 + "\n")
