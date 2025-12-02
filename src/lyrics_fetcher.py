"""
Lyrics fetching with multiple source fallbacks.

This module provides:
- Genius API integration (primary source)
- Genius song search for catalog expansion
- lyrics.ovh API fallback
- Lyrics validation and cleaning
"""

import logging
import os
import time
from typing import Optional, List, Dict

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LyricsFetcher:
    """
    Fetch song lyrics from multiple sources with fallback support.

    Sources (in priority order):
    1. Genius API (via lyricsgenius library) - primary
    2. lyrics.ovh API - fallback

    Enforces minimum lyric length requirement for HarmonyAI (40 characters).
    """

    def __init__(
        self,
        genius_token: Optional[str] = None,
        min_lyric_length: int = 40,
        genius_delay: float = 0.2,
        fallback_delay: float = 0.1
    ):
        """
        Initialize lyrics fetcher.

        Args:
            genius_token: Genius API access token (from environment if not provided)
            min_lyric_length: Minimum valid lyric length in characters (default: 40)
            genius_delay: Delay between Genius requests in seconds (default: 0.2)
            fallback_delay: Delay between fallback requests in seconds (default: 0.1)
        """
        self.min_lyric_length = min_lyric_length
        self.genius_delay = genius_delay
        self.fallback_delay = fallback_delay

        # Initialize Genius client
        self.genius_client = None
        token = genius_token or os.getenv('GENIUS_API_TOKEN')

        if token:
            try:
                import lyricsgenius
                self.genius_client = lyricsgenius.Genius(
                    access_token=token,
                    sleep_time=genius_delay,
                    timeout=15,
                    remove_section_headers=True,  # Clean [Verse], [Chorus] headers
                    skip_non_songs=True,  # Skip non-song results
                    verbose=False  # Reduce logging noise
                )
                logger.info("Genius API client initialized successfully")
            except ImportError:
                logger.warning("lyricsgenius not installed. Install with: pip install lyricsgenius")
            except Exception as e:
                logger.warning(f"Failed to initialize Genius client: {e}")
        else:
            logger.warning("No Genius API token provided. Set GENIUS_API_TOKEN environment variable.")

    def _clean_lyrics(self, lyrics: str) -> str:
        """
        Clean and normalize lyrics text.

        Args:
            lyrics: Raw lyrics string

        Returns:
            Cleaned lyrics string
        """
        if not lyrics:
            return ""

        # Strip whitespace
        lyrics = lyrics.strip()

        # Remove common artifacts from Genius scraping
        lyrics = lyrics.replace('\\n', '\n')

        # Remove multiple consecutive newlines
        while '\n\n\n' in lyrics:
            lyrics = lyrics.replace('\n\n\n', '\n\n')

        # Remove embed footer that Genius adds
        if 'Embed' in lyrics and len(lyrics) - lyrics.rfind('Embed') < 10:
            lyrics = lyrics[:lyrics.rfind('Embed')].strip()

        # Remove trailing numbers (often page numbers or annotations)
        lines = lyrics.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.isdigit():  # Skip lines that are just numbers
                cleaned_lines.append(line)

        lyrics = '\n'.join(cleaned_lines)

        return lyrics.strip()

    def _fetch_from_genius(self, title: str, artist: str) -> Optional[str]:
        """
        Fetch lyrics from Genius API.

        Args:
            title: Song title
            artist: Artist name

        Returns:
            Lyrics string if found and valid, None otherwise
        """
        if not self.genius_client:
            return None

        try:
            logger.debug(f"Searching Genius for: {artist} - {title}")

            song = self.genius_client.search_song(title, artist)

            if song and song.lyrics:
                lyrics = self._clean_lyrics(song.lyrics)

                if len(lyrics) >= self.min_lyric_length:
                    logger.debug(f"Found lyrics from Genius ({len(lyrics)} chars)")
                    return lyrics
                else:
                    logger.debug(f"Lyrics too short from Genius ({len(lyrics)} chars)")
                    return None

        except Exception as e:
            logger.debug(f"Genius fetch failed: {e}")

        return None

    def _fetch_from_lyrics_ovh(self, title: str, artist: str) -> Optional[str]:
        """
        Fetch lyrics from lyrics.ovh API (fallback).

        Args:
            title: Song title
            artist: Artist name

        Returns:
            Lyrics string if found and valid, None otherwise
        """
        try:
            logger.debug(f"Searching lyrics.ovh for: {artist} - {title}")

            # Add delay for courtesy (no strict rate limit)
            time.sleep(self.fallback_delay)

            url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
            response = requests.get(url, timeout=10)

            if response.ok:
                data = response.json()
                if 'lyrics' in data:
                    lyrics = self._clean_lyrics(data['lyrics'])

                    if len(lyrics) >= self.min_lyric_length:
                        logger.debug(f"Found lyrics from lyrics.ovh ({len(lyrics)} chars)")
                        return lyrics
                    else:
                        logger.debug(f"Lyrics too short from lyrics.ovh ({len(lyrics)} chars)")
                        return None

        except requests.RequestException as e:
            logger.debug(f"lyrics.ovh fetch failed: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error fetching from lyrics.ovh: {e}")

        return None

    def fetch_lyrics(
        self,
        title: str,
        artist: str,
        use_fallback: bool = True
    ) -> Optional[str]:
        """
        Fetch lyrics with automatic fallback.

        Tries Genius first, then falls back to lyrics.ovh if enabled.

        Args:
            title: Song title
            artist: Artist name
            use_fallback: Whether to use fallback source if primary fails (default: True)

        Returns:
            Cleaned lyrics string if found and meets minimum length, None otherwise
        """
        # Try Genius first (primary source)
        lyrics = self._fetch_from_genius(title, artist)

        if lyrics:
            logger.info(f"✓ Lyrics found [Genius]: {artist} - {title} ({len(lyrics)} chars)")
            return lyrics

        # Try fallback if enabled
        if use_fallback:
            lyrics = self._fetch_from_lyrics_ovh(title, artist)

            if lyrics:
                logger.info(f"✓ Lyrics found [lyrics.ovh]: {artist} - {title} ({len(lyrics)} chars)")
                return lyrics

        logger.warning(f"✗ No lyrics found: {artist} - {title}")
        return None

    def validate_lyrics(self, lyrics: Optional[str]) -> bool:
        """
        Validate that lyrics meet minimum requirements.

        Args:
            lyrics: Lyrics string to validate

        Returns:
            True if valid, False otherwise
        """
        if not lyrics:
            return False

        cleaned = self._clean_lyrics(lyrics)
        return len(cleaned) >= self.min_lyric_length

    def search_songs(
        self,
        query: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Search for songs on Genius API.

        This expands the catalog beyond MusicBrainz by searching
        Genius directly for songs matching a query.

        Args:
            query: Search query (artist, genre, mood, etc.)
            limit: Maximum number of songs to return (default: 50)

        Returns:
            List of song dictionaries with keys:
            - title: Song title
            - artist: Artist name
            - genius_id: Genius song ID
        """
        if not self.genius_client:
            logger.warning("Genius client not initialized. Cannot search songs.")
            return []

        try:
            logger.info(f"Searching Genius for: '{query}' (limit={limit})")

            # Use Genius search_songs method (searches across all songs)
            search_results = []
            page = 1
            per_page = min(limit, 50)  # Genius API returns max 50 per page

            while len(search_results) < limit:
                try:
                    # Search using the Genius API
                    results = self.genius_client.search_all(
                        search_term=query,
                        per_page=per_page,
                        page=page
                    )

                    if not results or 'sections' not in results:
                        break

                    # Extract songs from search results
                    for section in results['sections']:
                        if section['type'] == 'song':
                            for hit in section.get('hits', []):
                                if len(search_results) >= limit:
                                    break

                                result = hit.get('result', {})
                                title = result.get('title', '').strip()
                                artist_name = result.get('primary_artist', {}).get('name', 'Unknown').strip()
                                genius_id = result.get('id')

                                if title and artist_name and genius_id:
                                    search_results.append({
                                        'title': title,
                                        'artist': artist_name,
                                        'genius_id': genius_id
                                    })

                    # If we got fewer results than requested, we've reached the end
                    if not results.get('sections') or len(results['sections']) == 0:
                        break

                    page += 1
                    time.sleep(self.genius_delay)  # Rate limiting

                except Exception as e:
                    logger.debug(f"Error in page {page} of search: {e}")
                    break

            logger.info(f"Found {len(search_results)} songs on Genius for '{query}'")
            return search_results[:limit]

        except AttributeError:
            # Fallback: use simpler search if search_all not available
            logger.debug("Trying simpler Genius search method")
            try:
                songs = []
                # Try direct API call
                token = os.getenv('GENIUS_API_TOKEN')
                if not token:
                    return []

                url = f"https://api.genius.com/search?q={query}"
                headers = {'Authorization': f'Bearer {token}'}

                for page in range(1, (limit // 10) + 2):  # Each page returns ~10 results
                    response = requests.get(
                        url,
                        headers=headers,
                        params={'page': page, 'per_page': 10},
                        timeout=10
                    )

                    if response.ok:
                        data = response.json()
                        hits = data.get('response', {}).get('hits', [])

                        if not hits:
                            break

                        for hit in hits:
                            if len(songs) >= limit:
                                break

                            result = hit.get('result', {})
                            title = result.get('title', '').strip()
                            artist_name = result.get('primary_artist', {}).get('name', 'Unknown').strip()
                            genius_id = result.get('id')

                            if title and artist_name and genius_id:
                                songs.append({
                                    'title': title,
                                    'artist': artist_name,
                                    'genius_id': genius_id
                                })

                        time.sleep(self.genius_delay)
                    else:
                        break

                logger.info(f"Found {len(songs)} songs via direct API")
                return songs

            except Exception as e:
                logger.error(f"Genius song search failed: {e}")
                return []

        except Exception as e:
            logger.error(f"Genius song search failed: {e}")
            return []
