"""
Unit tests for lyrics fetcher module.

Tests cover:
- Lyrics cleaning and validation
- Genius API fallback chain
- Minimum length enforcement
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lyrics_fetcher import LyricsFetcher


class TestLyricsFetcher:
    """Test suite for LyricsFetcher class."""

    def test_clean_lyrics_removes_whitespace(self):
        """Test that lyrics cleaning removes excess whitespace."""
        fetcher = LyricsFetcher()

        raw_lyrics = "  \n\n  Test lyrics here  \n\n\n  "
        cleaned = fetcher._clean_lyrics(raw_lyrics)

        assert cleaned == "Test lyrics here"
        assert '\n\n\n' not in cleaned

    def test_clean_lyrics_removes_embed_footer(self):
        """Test that Genius 'Embed' footer is removed."""
        fetcher = LyricsFetcher()

        raw_lyrics = "These are song lyrics\nWith multiple lines\n123Embed"
        cleaned = fetcher._clean_lyrics(raw_lyrics)

        assert 'Embed' not in cleaned
        assert 'These are song lyrics' in cleaned

    def test_clean_lyrics_removes_numeric_lines(self):
        """Test that numeric-only lines are removed."""
        fetcher = LyricsFetcher()

        raw_lyrics = "Verse one\n123\nVerse two\n456\nChorus"
        cleaned = fetcher._clean_lyrics(raw_lyrics)

        lines = cleaned.split('\n')
        assert '123' not in lines
        assert '456' not in lines
        assert 'Verse one' in cleaned
        assert 'Chorus' in cleaned

    def test_validate_lyrics_enforces_minimum_length(self):
        """Test that lyrics validation enforces minimum length."""
        fetcher = LyricsFetcher(min_lyric_length=40)

        valid_lyrics = "These are valid lyrics with enough characters to pass the minimum length requirement"
        short_lyrics = "Too short"

        assert fetcher.validate_lyrics(valid_lyrics) is True
        assert fetcher.validate_lyrics(short_lyrics) is False
        assert fetcher.validate_lyrics(None) is False
        assert fetcher.validate_lyrics("") is False

    @patch('src.lyrics_fetcher.lyricsgenius.Genius')
    def test_fetch_from_genius_success(self, mock_genius_class):
        """Test successful lyrics fetch from Genius."""
        # Mock Genius client
        mock_genius = Mock()
        mock_song = Mock()
        mock_song.lyrics = "These are the complete lyrics from Genius API with enough length to pass validation checks"
        mock_genius.search_song.return_value = mock_song
        mock_genius_class.return_value = mock_genius

        fetcher = LyricsFetcher(genius_token="test_token")
        lyrics = fetcher._fetch_from_genius("Hey Jude", "The Beatles")

        assert lyrics is not None
        assert len(lyrics) >= 40
        assert "complete lyrics" in lyrics

    @patch('src.lyrics_fetcher.lyricsgenius.Genius')
    def test_fetch_from_genius_too_short(self, mock_genius_class):
        """Test that short lyrics from Genius are rejected."""
        mock_genius = Mock()
        mock_song = Mock()
        mock_song.lyrics = "Short"  # Only 5 chars
        mock_genius.search_song.return_value = mock_song
        mock_genius_class.return_value = mock_genius

        fetcher = LyricsFetcher(genius_token="test_token", min_lyric_length=40)
        lyrics = fetcher._fetch_from_genius("Test", "Artist")

        assert lyrics is None  # Should reject too-short lyrics

    @patch('src.lyrics_fetcher.requests.get')
    def test_fetch_from_lyrics_ovh_success(self, mock_get):
        """Test successful lyrics fetch from lyrics.ovh."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'lyrics': "These are lyrics from the lyrics.ovh API with sufficient length to meet requirements"
        }
        mock_get.return_value = mock_response

        fetcher = LyricsFetcher()
        lyrics = fetcher._fetch_from_lyrics_ovh("Song", "Artist")

        assert lyrics is not None
        assert len(lyrics) >= 40

    @patch('src.lyrics_fetcher.requests.get')
    def test_fetch_from_lyrics_ovh_failed(self, mock_get):
        """Test failed lyrics fetch from lyrics.ovh."""
        mock_response = Mock()
        mock_response.ok = False
        mock_get.return_value = mock_response

        fetcher = LyricsFetcher()
        lyrics = fetcher._fetch_from_lyrics_ovh("Nonexistent", "Artist")

        assert lyrics is None

    @patch('src.lyrics_fetcher.LyricsFetcher._fetch_from_genius')
    @patch('src.lyrics_fetcher.LyricsFetcher._fetch_from_lyrics_ovh')
    def test_fetch_lyrics_uses_fallback(self, mock_ovh, mock_genius):
        """Test that fallback is used when primary source fails."""
        # Genius fails
        mock_genius.return_value = None

        # lyrics.ovh succeeds
        mock_ovh.return_value = "Lyrics from fallback source with enough characters to be valid"

        fetcher = LyricsFetcher(genius_token="test_token")
        lyrics = fetcher.fetch_lyrics("Song", "Artist", use_fallback=True)

        assert lyrics is not None
        assert "fallback" in lyrics

        # Verify both sources were tried
        mock_genius.assert_called_once()
        mock_ovh.assert_called_once()

    @patch('src.lyrics_fetcher.LyricsFetcher._fetch_from_genius')
    @patch('src.lyrics_fetcher.LyricsFetcher._fetch_from_lyrics_ovh')
    def test_fetch_lyrics_skips_fallback_when_disabled(self, mock_ovh, mock_genius):
        """Test that fallback is skipped when disabled."""
        mock_genius.return_value = None

        fetcher = LyricsFetcher(genius_token="test_token")
        lyrics = fetcher.fetch_lyrics("Song", "Artist", use_fallback=False)

        assert lyrics is None

        # Verify only primary source was tried
        mock_genius.assert_called_once()
        mock_ovh.assert_not_called()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
