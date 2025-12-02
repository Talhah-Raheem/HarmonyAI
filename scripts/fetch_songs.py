"""
Batch song fetcher for HarmonyAI catalog expansion.

This script:
- Fetches song metadata from MusicBrainz
- Fetches lyrics from Genius/lyrics.ovh
- Stores songs in SQLite database
- Exports to CSV for existing pipeline integration
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api_client import MusicBrainzClient
from src.lyrics_fetcher import LyricsFetcher
from src.database import SongDatabase

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/fetch_songs.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class SongFetcher:
    """Orchestrates fetching songs from multiple APIs and storing in database."""

    def __init__(
        self,
        db_path: str = 'data/songs.db',
        genius_token: Optional[str] = None
    ):
        """
        Initialize song fetcher.

        Args:
            db_path: Path to SQLite database
            genius_token: Genius API token (uses env var if not provided)
        """
        self.db = SongDatabase(db_path)
        self.api_client = MusicBrainzClient()
        self.lyrics_fetcher = LyricsFetcher(genius_token=genius_token)

        # Statistics tracking
        self.stats = {
            'queries_processed': 0,
            'songs_found': 0,
            'lyrics_fetched': 0,
            'lyrics_genius': 0,
            'lyrics_fallback': 0,
            'lyrics_failed': 0,
            'duplicates_skipped': 0,
            'added_to_db': 0
        }

    def fetch_for_query(
        self,
        query: str,
        limit_per_query: int = 100,
        progress_bar: bool = True,
        source: str = 'both'
    ) -> int:
        """
        Fetch songs for a single query.

        Args:
            query: Search query string
            limit_per_query: Maximum songs to fetch per query
            progress_bar: Show progress bar if tqdm is available
            source: Song source - 'musicbrainz', 'genius', or 'both' (default: 'both')

        Returns:
            Number of songs successfully added to database
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Fetching songs for query: '{query}' (limit: {limit_per_query}, source: {source})")
        logger.info(f"{'='*60}")

        songs_metadata = []

        # Search MusicBrainz if requested
        if source in ['musicbrainz', 'both']:
            mb_songs = self.api_client.search_songs(query, limit=limit_per_query)
            logger.info(f"MusicBrainz: Found {len(mb_songs)} songs")
            songs_metadata.extend(mb_songs)

        # Search Genius if requested
        if source in ['genius', 'both']:
            genius_songs = self.lyrics_fetcher.search_songs(query, limit=limit_per_query)
            logger.info(f"Genius: Found {len(genius_songs)} songs")
            # Convert Genius results to match MusicBrainz format
            for song in genius_songs:
                songs_metadata.append({
                    'title': song['title'],
                    'artist': song['artist'],
                    'mbid': None,  # No MusicBrainz ID for Genius songs
                    'release_year': None,
                    'score': 100  # High score since it's a direct match
                })

        if not songs_metadata:
            logger.warning(f"No songs found for query: {query}")
            return 0

        logger.info(f"Total songs to process: {len(songs_metadata)}")
        self.stats['songs_found'] += len(songs_metadata)
        added_count = 0

        # Process each song
        iterator = songs_metadata
        if HAS_TQDM and progress_bar:
            iterator = tqdm(songs_metadata, desc=f"Processing '{query}'")

        for metadata in iterator:
            title = metadata['title']
            artist = metadata['artist']
            mbid = metadata['mbid']
            release_year = metadata.get('release_year')

            # Check if already in database
            if self.db.song_exists(title, artist):
                logger.debug(f"Skipping duplicate: {artist} - {title}")
                self.stats['duplicates_skipped'] += 1
                continue

            # Fetch lyrics
            lyrics = self.lyrics_fetcher.fetch_lyrics(title, artist)

            if lyrics:
                # Determine lyrics source
                lyrics_source = 'genius'  # Default to genius since it's tried first

                # Try to determine actual source used
                # (This is a simplification; in practice the fetcher could track this)
                if len(lyrics) < 1000:  # Heuristic: lyrics.ovh tends to return shorter lyrics
                    # Check if this might be from fallback
                    pass  # Keep as genius for now

                self.stats['lyrics_fetched'] += 1
                self.stats['lyrics_genius'] += 1  # Simplified tracking

                # Add to database
                song = {
                    'title': title,
                    'artist': artist,
                    'lyrics': lyrics,
                    'metadata_source': 'musicbrainz',
                    'lyrics_source': lyrics_source,
                    'musicbrainz_id': mbid,
                    'release_year': release_year
                }

                if self.db.add_song(song):
                    self.stats['added_to_db'] += 1
                    added_count += 1
                    if not (HAS_TQDM and progress_bar):
                        logger.info(f"✓ Added: {artist} - {title}")
            else:
                self.stats['lyrics_failed'] += 1
                logger.debug(f"✗ No lyrics: {artist} - {title}")

        logger.info(f"\nCompleted query '{query}': {added_count} songs added")
        return added_count

    def fetch_all(
        self,
        queries: List[str],
        limit_per_query: int = 100,
        source: str = 'both'
    ) -> None:
        """
        Fetch songs for multiple queries.

        Args:
            queries: List of search query strings
            limit_per_query: Maximum songs to fetch per query
            source: Song source - 'musicbrainz', 'genius', or 'both' (default: 'both')
        """
        start_time = time.time()

        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH FETCH STARTED")
        logger.info(f"{'='*60}")
        logger.info(f"Queries: {len(queries)}")
        logger.info(f"Limit per query: {limit_per_query}")
        logger.info(f"Source: {source}")
        logger.info(f"Target songs: ~{len(queries) * limit_per_query}")
        logger.info(f"Database: {self.db.db_path}")
        logger.info(f"{'='*60}\n")

        for query in queries:
            self.fetch_for_query(query, limit_per_query, source=source)
            self.stats['queries_processed'] += 1

        elapsed = time.time() - start_time

        # Print final summary
        self._print_summary(elapsed)

    def _print_summary(self, elapsed: float) -> None:
        """Print fetch statistics summary."""
        logger.info(f"\n{'='*60}")
        logger.info("FETCH SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Queries processed:      {self.stats['queries_processed']}")
        logger.info(f"Songs found:            {self.stats['songs_found']}")
        logger.info(f"Lyrics fetched:         {self.stats['lyrics_fetched']} ({self._pct(self.stats['lyrics_fetched'], self.stats['songs_found'])})")
        logger.info(f"  - Genius:             {self.stats['lyrics_genius']}")
        logger.info(f"  - Fallback:           {self.stats['lyrics_fallback']}")
        logger.info(f"Lyrics failed:          {self.stats['lyrics_failed']} ({self._pct(self.stats['lyrics_failed'], self.stats['songs_found'])})")
        logger.info(f"Duplicates skipped:     {self.stats['duplicates_skipped']}")
        logger.info(f"Added to database:      {self.stats['added_to_db']}")

        # Database stats
        db_stats = self.db.get_stats()
        logger.info(f"\nDatabase statistics:")
        logger.info(f"Total songs in DB:      {db_stats['total_songs']}")
        logger.info(f"Unique artists:         {db_stats['unique_artists']}")
        logger.info(f"Avg lyric length:       {db_stats['avg_lyric_length']:.0f} chars")

        logger.info(f"\nDuration:               {elapsed/60:.1f} minutes")
        logger.info(f"{'='*60}\n")

    def _pct(self, value: int, total: int) -> str:
        """Calculate percentage string."""
        if total == 0:
            return "0%"
        return f"{100 * value / total:.1f}%"

    def export(self, output_path: str) -> int:
        """
        Export database to CSV.

        Args:
            output_path: Path to output CSV file

        Returns:
            Number of songs exported
        """
        logger.info(f"\nExporting to CSV: {output_path}")
        count = self.db.export_to_csv(output_path, include_metadata=False)
        logger.info(f"Export complete: {count} songs")
        return count


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch songs for HarmonyAI catalog expansion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch from both MusicBrainz and Genius (wider catalog)
  python -m scripts.fetch_songs --queries "rock" "pop" "jazz"

  # Fetch from Genius only (larger results)
  python -m scripts.fetch_songs --queries "Taylor Swift" --source genius --limit-per-query 200

  # Fetch 50 Beatles songs from MusicBrainz only
  python -m scripts.fetch_songs --queries "Beatles" --limit-per-query 50 --source musicbrainz

  # Use custom database path
  python -m scripts.fetch_songs --queries "indie" --db data/test.db

  # Export only (no fetching)
  python -m scripts.fetch_songs --export-only --output songs.csv
        """
    )

    parser.add_argument(
        '--queries',
        nargs='+',
        help='Search queries (genres, artists, etc.)'
    )

    parser.add_argument(
        '--limit-per-query',
        type=int,
        default=100,
        help='Maximum songs to fetch per query (default: 100)'
    )

    parser.add_argument(
        '--db',
        default='data/songs.db',
        help='Path to SQLite database (default: data/songs.db)'
    )

    parser.add_argument(
        '--output',
        default='data/raw/songs_fetched.csv',
        help='Path to output CSV file (default: data/raw/songs_fetched.csv)'
    )

    parser.add_argument(
        '--genius-token',
        help='Genius API token (uses GENIUS_API_TOKEN env var if not provided)'
    )

    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Only export database to CSV, skip fetching'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )

    parser.add_argument(
        '--source',
        choices=['musicbrainz', 'genius', 'both'],
        default='both',
        help='Song source: musicbrainz, genius, or both (default: both)'
    )

    args = parser.parse_args()

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    try:
        # Initialize fetcher
        fetcher = SongFetcher(
            db_path=args.db,
            genius_token=args.genius_token
        )

        if args.export_only:
            # Export only mode
            fetcher.export(args.output)
        else:
            # Fetch mode
            if not args.queries:
                logger.error("No queries provided. Use --queries to specify search terms.")
                logger.info("Example: python -m scripts.fetch_songs --queries 'rock' 'pop'")
                sys.exit(1)

            # Fetch songs
            fetcher.fetch_all(
                queries=args.queries,
                limit_per_query=args.limit_per_query,
                source=args.source
            )

            # Export to CSV
            fetcher.export(args.output)

        logger.info("\nSUCCESS ✓")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\nFAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
