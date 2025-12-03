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
        Search for songs by ARTIST using Genius API.
        """
        if not self.genius_client:
            logger.warning("Genius client not initialized.")
            return []

        try:
            logger.info(f"Fetching top {limit} songs for artist: '{query}'")

            # Use search_artist to reliably get songs by this artist
            # We use get_full_info=False to make it faster (we fetch lyrics later)
            artist = self.genius_client.search_artist(
                query,
                max_songs=limit,
                sort='popularity',
                get_full_info=False
            )

            if not artist:
                logger.warning(f"Artist not found: {query}")
                return []

            songs = []
            for song in artist.songs:
                songs.append({
                    'title': song.title,
                    'artist': artist.name,
                })

            logger.info(f"Found {len(songs)} songs for {artist.name}")
            return songs

        except Exception as e:
            logger.error(f"Genius artist search failed: {e}")
            return []
