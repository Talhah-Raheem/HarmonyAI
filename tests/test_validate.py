"""
Unit tests for validate module.

Tests cover:
- Schema validation with missing columns
- Schema validation with valid data
- Basic statistics computation
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validate import validate_schema, basic_stats


class TestValidateSchema:
    """Test suite for validate_schema function."""

    def test_raises_error_on_missing_columns(self):
        """Test that ValueError is raised when required columns are missing."""
        # Missing 'lyrics' column
        df = pd.DataFrame({
            'song_id': [1, 2],
            'title': ['Song A', 'Song B'],
            'artist': ['Artist 1', 'Artist 2']
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_raises_error_on_empty_dataframe(self):
        """Test that ValueError is raised for empty DataFrame."""
        df = pd.DataFrame({
            'song_id': [],
            'title': [],
            'artist': [],
            'lyrics': []
        })

        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_schema(df)

    def test_raises_error_on_wrong_song_id_type(self):
        """Test that ValueError is raised when song_id is not integer."""
        df = pd.DataFrame({
            'song_id': ['a', 'b', 'c'],  # String instead of int
            'title': ['Song A', 'Song B', 'Song C'],
            'artist': ['Artist 1', 'Artist 2', 'Artist 3'],
            'lyrics': ['Lyrics A', 'Lyrics B', 'Lyrics C']
        })

        with pytest.raises(ValueError, match="song_id must be integer"):
            validate_schema(df)

    def test_passes_with_valid_schema(self):
        """Test that validation passes with valid schema."""
        df = pd.DataFrame({
            'song_id': [1, 2, 3],
            'title': ['Song A', 'Song B', 'Song C'],
            'artist': ['Artist 1', 'Artist 2', 'Artist 3'],
            'lyrics': [
                'These are lyrics for song A with enough content',
                'These are lyrics for song B with enough content',
                'These are lyrics for song C with enough content'
            ]
        })

        # Should not raise any exception
        validate_schema(df)

    def test_allows_extra_columns(self):
        """Test that extra columns are allowed."""
        df = pd.DataFrame({
            'song_id': [1, 2],
            'title': ['Song A', 'Song B'],
            'artist': ['Artist 1', 'Artist 2'],
            'lyrics': ['Lyrics A with content', 'Lyrics B with content'],
            'extra_column': ['Extra 1', 'Extra 2']  # Extra column
        })

        # Should not raise any exception
        validate_schema(df)


class TestBasicStats:
    """Test suite for basic_stats function."""

    def test_computes_correct_statistics(self):
        """Test that statistics are computed correctly."""
        df = pd.DataFrame({
            'song_id': [1, 2, 3],
            'title': ['Song A', 'Song B', 'Song C'],
            'artist': ['Artist 1', 'Artist 2', 'Artist 1'],  # Artist 1 appears twice
            'lyrics': [
                'Short',  # 5 chars
                'Medium length lyrics',  # 20 chars
                'This is a much longer set of lyrics for testing'  # 47 chars
            ]
        })

        stats = basic_stats(df)

        assert stats['row_count'] == 3
        assert stats['unique_artists'] == 2  # Artist 1 and Artist 2
        assert stats['unique_songs'] == 3
        assert stats['lyric_length_min'] == 5
        assert stats['lyric_length_max'] == 47
        assert 'lyric_length_mean' in stats
        assert 'lyric_length_median' in stats

    def test_handles_null_values(self):
        """Test that null values are counted correctly."""
        df = pd.DataFrame({
            'song_id': [1, 2, 3],
            'title': ['Song A', None, 'Song C'],
            'artist': ['Artist 1', 'Artist 2', None],
            'lyrics': ['Lyrics A', 'Lyrics B', 'Lyrics C']
        })

        stats = basic_stats(df)

        assert stats['null_counts']['title'] == 1
        assert stats['null_counts']['artist'] == 1
        assert stats['null_counts']['lyrics'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
