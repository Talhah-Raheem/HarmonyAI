"""
Unit tests for data_prep module.

Tests cover:
- Short lyric removal
- Duplicate removal
- Empty lyric removal
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import clean_songs


class TestCleanSongs:
    """Test suite for clean_songs function."""

    def test_removes_short_lyrics(self):
        """Test that songs with lyrics below minimum length are removed."""
        df = pd.DataFrame({
            'song_id': [1, 2, 3],
            'title': ['Song A', 'Song B', 'Song C'],
            'artist': ['Artist 1', 'Artist 2', 'Artist 3'],
            'lyrics': [
                'Short',  # Only 5 chars
                'This is a longer lyric that meets the minimum character requirement',
                'Another long lyric with enough characters to pass the filter'
            ]
        })

        result = clean_songs(df, min_lyric_chars=40)

        # Should only keep songs with lyrics >= 40 chars
        assert len(result) == 2
        assert 'Song A' not in result['title'].values
        assert 'Song B' in result['title'].values
        assert 'Song C' in result['title'].values

    def test_removes_duplicates(self):
        """Test that duplicate songs (by title and artist) are removed."""
        df = pd.DataFrame({
            'song_id': [1, 2, 3, 4],
            'title': ['Imagine', 'Imagine', 'Yesterday', 'Hey Jude'],
            'artist': ['John Lennon', 'John Lennon', 'The Beatles', 'The Beatles'],
            'lyrics': [
                'Imagine all the people living life in peace with love and harmony',
                'Imagine all the people living life in peace with love and harmony',
                'Yesterday all my troubles seemed so far away from here today',
                'Hey Jude dont make it bad take a sad song and make it better'
            ]
        })

        result = clean_songs(df, min_lyric_chars=40)

        # Should remove one duplicate 'Imagine' by 'John Lennon'
        assert len(result) == 3
        assert result['title'].value_counts()['Imagine'] == 1

    def test_removes_empty_lyrics(self):
        """Test that songs with empty or null lyrics are removed."""
        df = pd.DataFrame({
            'song_id': [1, 2, 3, 4],
            'title': ['Song A', 'Song B', 'Song C', 'Song D'],
            'artist': ['Artist 1', 'Artist 2', 'Artist 3', 'Artist 4'],
            'lyrics': [
                'Valid lyrics with enough characters to pass the minimum requirement',
                '',  # Empty string
                None,  # Null value
                '   '  # Only whitespace
            ]
        })

        result = clean_songs(df, min_lyric_chars=40)

        # Should only keep Song A
        assert len(result) == 1
        assert result.iloc[0]['title'] == 'Song A'

    def test_case_insensitive_deduplication(self):
        """Test that deduplication is case-insensitive."""
        df = pd.DataFrame({
            'song_id': [1, 2, 3],
            'title': ['Imagine', 'IMAGINE', 'imagine'],
            'artist': ['John Lennon', 'JOHN LENNON', 'john lennon'],
            'lyrics': [
                'Imagine all the people living life in peace with love and harmony',
                'Imagine all the people living life in peace with love and harmony',
                'Imagine all the people living life in peace with love and harmony'
            ]
        })

        result = clean_songs(df, min_lyric_chars=40)

        # Should only keep one song (case-insensitive match)
        assert len(result) == 1

    def test_preserves_valid_data(self):
        """Test that valid data is preserved correctly."""
        df = pd.DataFrame({
            'song_id': [1, 2],
            'title': ['Song A', 'Song B'],
            'artist': ['Artist 1', 'Artist 2'],
            'lyrics': [
                'These are valid lyrics with enough characters to meet requirements',
                'Another set of valid lyrics that should be kept in the dataset'
            ]
        })

        result = clean_songs(df, min_lyric_chars=40)

        # Should keep all valid songs
        assert len(result) == 2
        assert list(result['title']) == ['Song A', 'Song B']
        assert list(result['artist']) == ['Artist 1', 'Artist 2']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
