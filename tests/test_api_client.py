"""
Unit tests for API client module.

Tests cover:
- Rate limiter functionality
- MusicBrainz API integration
- Error handling and retries
"""

import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api_client import RateLimiter, MusicBrainzClient


class TestRateLimiter:
    """Test suite for RateLimiter class."""

    def test_rate_limiter_respects_interval(self):
        """Test that rate limiter enforces minimum interval."""
        limiter = RateLimiter(min_interval=0.1)  # 100ms

        start = time.time()

        # First request - no delay
        limiter.wait_if_needed()
        first_elapsed = time.time() - start

        # Second request - should delay
        limiter.wait_if_needed()
        second_elapsed = time.time() - start

        # Should have waited at least min_interval
        assert second_elapsed >= 0.1
        assert first_elapsed < 0.05  # First request should be immediate

    def test_rate_limiter_no_delay_first_request(self):
        """Test that first request has no delay."""
        limiter = RateLimiter(min_interval=1.0)

        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start

        # First request should be essentially immediate
        assert elapsed < 0.1


class TestMusicBrainzClient:
    """Test suite for MusicBrainzClient class."""

    def test_client_initialization(self):
        """Test that client initializes with correct headers."""
        client = MusicBrainzClient(user_agent="TestAgent/1.0")

        assert client.user_agent == "TestAgent/1.0"
        assert client.session.headers['User-Agent'] == "TestAgent/1.0"
        assert client.session.headers['Accept'] == "application/json"

    @patch('src.api_client.requests.Session.get')
    def test_search_songs_success(self, mock_get):
        """Test successful song search."""
        # Mock API response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'recordings': [
                {
                    'id': 'test-mbid-123',
                    'title': 'Hey Jude',
                    'artist-credit': [{'name': 'The Beatles'}],
                    'first-release-date': '1968-08-26',
                    'score': 100
                }
            ]
        }
        mock_get.return_value = mock_response

        client = MusicBrainzClient(rate_limit=0)  # Disable rate limiting for tests
        results = client.search_songs("Beatles", limit=10)

        assert len(results) == 1
        assert results[0]['title'] == 'Hey Jude'
        assert results[0]['artist'] == 'The Beatles'
        assert results[0]['mbid'] == 'test-mbid-123'
        assert results[0]['release_year'] == 1968

    @patch('src.api_client.requests.Session.get')
    def test_search_songs_no_results(self, mock_get):
        """Test search with no results."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {'recordings': []}
        mock_get.return_value = mock_response

        client = MusicBrainzClient(rate_limit=0)
        results = client.search_songs("NonexistentArtist123456", limit=10)

        assert len(results) == 0

    @patch('src.api_client.requests.Session.get')
    def test_search_songs_handles_missing_fields(self, mock_get):
        """Test that search handles recordings with missing fields gracefully."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'recordings': [
                {
                    'id': 'mbid-1',
                    'title': 'Complete Song',
                    'artist-credit': [{'name': 'Artist One'}],
                    'score': 90
                },
                {
                    'id': None,  # Missing ID - should be skipped
                    'title': 'Incomplete Song',
                    'artist-credit': [{'name': 'Artist Two'}]
                },
                {
                    'id': 'mbid-3',
                    'title': '',  # Empty title - should be skipped
                    'artist-credit': [{'name': 'Artist Three'}]
                }
            ]
        }
        mock_get.return_value = mock_response

        client = MusicBrainzClient(rate_limit=0)
        results = client.search_songs("test", limit=10)

        # Should only return the complete recording
        assert len(results) == 1
        assert results[0]['title'] == 'Complete Song'

    @patch('src.api_client.requests.Session.get')
    def test_rate_limit_retry(self, mock_get):
        """Test that client retries on rate limit (HTTP 429)."""
        import requests

        # First call returns 429, second succeeds
        response_429 = Mock()
        response_429.ok = False
        response_429.status_code = 429
        response_429.raise_for_status.side_effect = requests.HTTPError(response=response_429)

        response_success = Mock()
        response_success.ok = True
        response_success.json.return_value = {'recordings': []}

        mock_get.side_effect = [response_429, response_success]

        client = MusicBrainzClient(rate_limit=0, max_retries=3)

        with patch('time.sleep'):  # Skip actual sleep for faster tests
            results = client.search_songs("test", limit=10)

        # Should succeed on retry
        assert results == []
        assert mock_get.call_count == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
