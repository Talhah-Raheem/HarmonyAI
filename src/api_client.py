"""
MusicBrainz API client for fetching song metadata.

This module provides:
- Rate-limited API client for MusicBrainz
- Song metadata fetching with proper error handling
- Retry logic for failed requests
"""

import logging
import time
from typing import List, Dict, Optional
from datetime import datetime

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Enforce minimum time interval between requests.

    MusicBrainz requires 1 request/second minimum interval.
    """

    def __init__(self, min_interval: float = 1.0):
        """
        Initialize rate limiter.

        Args:
            min_interval: Minimum seconds between requests (default: 1.0)
        """
        self.min_interval = min_interval
        self.last_request_time: Optional[float] = None

    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limit."""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

        self.last_request_time = time.time()


class MusicBrainzClient:
    """
    Client for MusicBrainz API to fetch song metadata.

    MusicBrainz is a free, open music encyclopedia that provides
    song metadata, artist information, and release details.
    """

    BASE_URL = "https://musicbrainz.org/ws/2"

    def __init__(
        self,
        user_agent: str = "HarmonyAI/1.0 (class-project)",
        rate_limit: float = 1.0,
        max_retries: int = 3
    ):
        """
        Initialize MusicBrainz client.

        Args:
            user_agent: User-Agent string (required by MusicBrainz)
            rate_limit: Minimum seconds between requests (default: 1.0)
            max_retries: Maximum retry attempts for failed requests
        """
        self.user_agent = user_agent
        self.rate_limiter = RateLimiter(min_interval=rate_limit)
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'application/json'
        })

    def _make_request(
        self,
        endpoint: str,
        params: Dict,
        retry_count: int = 0
    ) -> Optional[Dict]:
        """
        Make rate-limited HTTP request with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            retry_count: Current retry attempt number

        Returns:
            JSON response as dictionary, or None if all retries failed
        """
        self.rate_limiter.wait_if_needed()

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit exceeded
                logger.warning("Rate limit exceeded (HTTP 429)")
                if retry_count < self.max_retries:
                    backoff = 2 ** retry_count  # Exponential backoff: 2s, 4s, 8s
                    logger.info(f"Retrying in {backoff}s... (attempt {retry_count + 1}/{self.max_retries})")
                    time.sleep(backoff)
                    return self._make_request(endpoint, params, retry_count + 1)
                else:
                    logger.error("Max retries exceeded for rate limiting")
                    return None
            else:
                logger.error(f"HTTP error: {e}")
                return None

        except requests.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            if retry_count < self.max_retries:
                backoff = 2 ** retry_count
                logger.info(f"Retrying in {backoff}s...")
                time.sleep(backoff)
                return self._make_request(endpoint, params, retry_count + 1)
            return None

        except requests.Timeout:
            logger.error("Request timeout")
            if retry_count < self.max_retries:
                time.sleep(2)
                return self._make_request(endpoint, params, retry_count + 1)
            return None

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def search_songs(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Search for songs (recordings) by query string.

        Args:
            query: Search query (e.g., "rock", "Beatles", "Yesterday")
            limit: Maximum number of results (default: 100, max: 100)
            offset: Offset for pagination (default: 0)

        Returns:
            List of song dictionaries with keys:
            - mbid: MusicBrainz ID
            - title: Song title
            - artist: Artist name
            - release_year: Year of first release (if available)
            - score: Relevance score from MusicBrainz (0-100)
        """
        # MusicBrainz limits to 100 results per request
        limit = min(limit, 100)

        params = {
            'query': query,
            'limit': limit,
            'offset': offset,
            'fmt': 'json'
        }

        logger.info(f"Searching MusicBrainz for: '{query}' (limit={limit})")

        response = self._make_request('recording', params)

        if not response or 'recordings' not in response:
            logger.warning(f"No results found for query: {query}")
            return []

        songs = []
        for recording in response['recordings']:
            # Extract basic info
            mbid = recording.get('id')
            title = recording.get('title', '').strip()

            # Extract artist name (first artist from artist-credit)
            artist = 'Unknown Artist'
            if 'artist-credit' in recording and recording['artist-credit']:
                artist_credit = recording['artist-credit'][0]
                if isinstance(artist_credit, dict) and 'name' in artist_credit:
                    artist = artist_credit['name'].strip()
                elif isinstance(artist_credit, dict) and 'artist' in artist_credit:
                    artist = artist_credit['artist'].get('name', 'Unknown Artist').strip()

            # Extract release year if available
            release_year = None
            if 'first-release-date' in recording:
                date_str = recording['first-release-date']
                if date_str and len(date_str) >= 4:
                    try:
                        release_year = int(date_str[:4])
                    except ValueError:
                        pass

            # Get relevance score
            score = recording.get('score', 0)

            # Skip if missing essential fields
            if not title or not mbid:
                continue

            songs.append({
                'mbid': mbid,
                'title': title,
                'artist': artist,
                'release_year': release_year,
                'score': score
            })

        logger.info(f"Found {len(songs)} recordings")
        return songs

    def get_recording_details(self, mbid: str) -> Optional[Dict]:
        """
        Get detailed information for a specific recording.

        Args:
            mbid: MusicBrainz ID

        Returns:
            Dictionary with detailed recording information, or None if not found
        """
        params = {
            'inc': 'artist-credits+releases',
            'fmt': 'json'
        }

        logger.debug(f"Fetching details for MBID: {mbid}")

        response = self._make_request(f'recording/{mbid}', params)

        if not response:
            logger.warning(f"No details found for MBID: {mbid}")
            return None

        return response
