"""
Unit tests for database module.

Tests cover:
- Song insertion and retrieval
- Duplicate prevention
- CSV export functionality
- Statistics computation
"""

import sys
import tempfile
from pathlib import Path

import pytest
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import SongDatabase


class TestSongDatabase:
    """Test suite for SongDatabase class."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        db = SongDatabase(db_path)
        yield db
        db.close()

        # Clean up
        Path(db_path).unlink(missing_ok=True)

    def test_add_song_success(self, temp_db):
        """Test successful song addition."""
        song = {
            'title': 'Test Song',
            'artist': 'Test Artist',
            'lyrics': 'These are test lyrics that meet the minimum length requirement of 40 characters'
        }

        result = temp_db.add_song(song)

        assert result is True
        assert temp_db.get_song_count() == 1

    def test_add_song_too_short_lyrics(self, temp_db):
        """Test that songs with short lyrics are rejected."""
        song = {
            'title': 'Short Song',
            'artist': 'Test Artist',
            'lyrics': 'Too short'  # Less than 40 chars
        }

        result = temp_db.add_song(song)

        assert result is False
        assert temp_db.get_song_count() == 0

    def test_add_song_duplicate(self, temp_db):
        """Test that duplicate songs are prevented."""
        song = {
            'title': 'Duplicate Song',
            'artist': 'Test Artist',
            'lyrics': 'These are lyrics with enough characters to pass the minimum length requirement'
        }

        # Add first time - should succeed
        result1 = temp_db.add_song(song)
        assert result1 is True

        # Add second time - should fail (duplicate)
        result2 = temp_db.add_song(song)
        assert result2 is False

        # Should only have one song
        assert temp_db.get_song_count() == 1

    def test_song_exists(self, temp_db):
        """Test song existence check."""
        song = {
            'title': 'Existing Song',
            'artist': 'Test Artist',
            'lyrics': 'These are lyrics with sufficient length to meet the minimum requirement'
        }

        # Should not exist initially
        assert temp_db.song_exists('Existing Song', 'Test Artist') is False

        # Add song
        temp_db.add_song(song)

        # Should exist now
        assert temp_db.song_exists('Existing Song', 'Test Artist') is True

    def test_get_song(self, temp_db):
        """Test retrieving a song."""
        song = {
            'title': 'Retrievable Song',
            'artist': 'Test Artist',
            'lyrics': 'These are lyrics that can be retrieved from the database successfully'
        }

        temp_db.add_song(song)

        retrieved = temp_db.get_song('Retrievable Song', 'Test Artist')

        assert retrieved is not None
        assert retrieved['title'] == 'Retrievable Song'
        assert retrieved['artist'] == 'Test Artist'
        assert len(retrieved['lyrics']) >= 40

    def test_get_song_not_found(self, temp_db):
        """Test retrieving non-existent song."""
        result = temp_db.get_song('Nonexistent', 'Unknown')

        assert result is None

    def test_get_song_count(self, temp_db):
        """Test song count."""
        assert temp_db.get_song_count() == 0

        # Add some songs
        for i in range(5):
            song = {
                'title': f'Song {i}',
                'artist': f'Artist {i}',
                'lyrics': f'These are the lyrics for song number {i} which are long enough to pass validation'
            }
            temp_db.add_song(song)

        assert temp_db.get_song_count() == 5

    def test_get_stats(self, temp_db):
        """Test statistics computation."""
        # Add multiple songs
        songs = [
            {
                'title': 'Song 1',
                'artist': 'Artist A',
                'lyrics': 'A' * 100,
                'lyrics_source': 'genius',
                'release_year': 2020
            },
            {
                'title': 'Song 2',
                'artist': 'Artist A',  # Same artist
                'lyrics': 'B' * 200,
                'lyrics_source': 'genius',
                'release_year': 2020
            },
            {
                'title': 'Song 3',
                'artist': 'Artist B',
                'lyrics': 'C' * 150,
                'lyrics_source': 'lyrics_ovh',
                'release_year': 2015
            }
        ]

        for song in songs:
            temp_db.add_song(song)

        stats = temp_db.get_stats()

        assert stats['total_songs'] == 3
        assert stats['unique_artists'] == 2  # Artist A and Artist B
        assert stats['avg_lyric_length'] > 0
        assert 'genius' in stats['by_source']
        assert 'lyrics_ovh' in stats['by_source']
        assert stats['by_source']['genius'] == 2
        assert stats['by_source']['lyrics_ovh'] == 1

    def test_export_to_csv(self, temp_db):
        """Test CSV export functionality."""
        # Add songs
        songs = [
            {
                'title': 'Export Song 1',
                'artist': 'Artist 1',
                'lyrics': 'These are the lyrics for the first song to be exported to CSV'
            },
            {
                'title': 'Export Song 2',
                'artist': 'Artist 2',
                'lyrics': 'These are the lyrics for the second song to be exported to CSV'
            }
        ]

        for song in songs:
            temp_db.add_song(song)

        # Export to temporary CSV
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = f.name

        try:
            count = temp_db.export_to_csv(csv_path, include_metadata=False)

            assert count == 2

            # Verify CSV contents
            df = pd.read_csv(csv_path)
            assert len(df) == 2
            assert 'song_id' in df.columns
            assert 'title' in df.columns
            assert 'artist' in df.columns
            assert 'lyrics' in df.columns

            # Should NOT include metadata columns when include_metadata=False
            assert 'musicbrainz_id' not in df.columns
            assert 'created_at' not in df.columns

        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_deterministic_song_id(self, temp_db):
        """Test that song IDs are deterministic."""
        song1 = {
            'title': 'Same Song',
            'artist': 'Same Artist',
            'lyrics': 'First instance of lyrics that are long enough to pass validation'
        }

        song2 = {
            'title': 'Same Song',
            'artist': 'Same Artist',
            'lyrics': 'Second instance of lyrics that are long enough to pass validation'
        }

        # Add first song
        temp_db.add_song(song1)
        retrieved1 = temp_db.get_song('Same Song', 'Same Artist')

        # Try to add duplicate (should fail)
        temp_db.add_song(song2)

        # Retrieve again
        retrieved2 = temp_db.get_song('Same Song', 'Same Artist')

        # Song IDs should be identical (deterministic)
        assert retrieved1['song_id'] == retrieved2['song_id']

        # Should match the expected deterministic ID
        expected_id = abs(hash(('Same Song' + 'Same Artist').lower())) % (10**9)
        assert retrieved1['song_id'] == expected_id


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
