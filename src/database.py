"""
SQLite database management for song catalog.

This module provides:
- SQLite schema for caching fetched songs
- CRUD operations for song management
- CSV export for backward compatibility with existing pipeline
- Duplicate prevention and data integrity
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SongDatabase:
    """
    SQLite database manager for HarmonyAI song catalog.

    Stores fetched songs with metadata and provides export to CSV
    for integration with existing data processing pipeline.
    """

    def __init__(self, db_path: str = 'data/songs.db'):
        """
        Initialize database connection and create schema if needed.

        Args:
            db_path: Path to SQLite database file (default: data/songs.db)
        """
        self.db_path = db_path

        # Create parent directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize connection
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

        # Create schema
        self._create_schema()

        logger.info(f"Database initialized: {db_path}")

    def _create_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        cursor = self.conn.cursor()

        # Main songs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                song_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                artist TEXT NOT NULL,
                lyrics TEXT NOT NULL CHECK(length(lyrics) >= 40),

                -- API source tracking
                metadata_source TEXT,
                lyrics_source TEXT,

                -- External IDs
                musicbrainz_id TEXT UNIQUE,

                -- Metadata
                release_year INTEGER,

                -- Timestamps
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- Ensure no duplicate (title, artist) pairs
                UNIQUE(title, artist)
            )
        """)

        # Create indexes for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artist ON songs(artist)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_title ON songs(title)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_release_year ON songs(release_year)
        """)

        self.conn.commit()
        logger.debug("Database schema created/verified")

    def _generate_song_id(self, title: str, artist: str) -> int:
        """
        Generate deterministic song ID from title and artist.

        Matches the ID generation logic in src/data_prep.py:
        song_id = abs(hash((title + artist).lower())) % (10**9)

        Args:
            title: Song title
            artist: Artist name

        Returns:
            Integer song ID
        """
        return abs(hash((title + artist).lower())) % (10**9)

    def add_song(self, song: Dict) -> bool:
        """
        Add a song to the database.

        Args:
            song: Dictionary with keys:
                - title (required)
                - artist (required)
                - lyrics (required, >= 40 chars)
                - metadata_source (optional)
                - lyrics_source (optional)
                - musicbrainz_id (optional)
                - release_year (optional)

        Returns:
            True if song was added, False if it already exists or failed
        """
        try:
            # Generate song_id
            song_id = self._generate_song_id(song['title'], song['artist'])

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO songs (
                    song_id, title, artist, lyrics,
                    metadata_source, lyrics_source,
                    musicbrainz_id, release_year
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                song_id,
                song['title'],
                song['artist'],
                song['lyrics'],
                song.get('metadata_source'),
                song.get('lyrics_source'),
                song.get('musicbrainz_id'),
                song.get('release_year')
            ))

            self.conn.commit()
            logger.debug(f"Added song: {song['artist']} - {song['title']}")
            return True

        except sqlite3.IntegrityError as e:
            if 'UNIQUE constraint failed' in str(e):
                logger.debug(f"Song already exists: {song['artist']} - {song['title']}")
            elif 'CHECK constraint failed' in str(e):
                logger.warning(f"Lyrics too short for: {song['artist']} - {song['title']}")
            else:
                logger.error(f"Integrity error: {e}")
            return False

        except KeyError as e:
            logger.error(f"Missing required field: {e}")
            return False

        except Exception as e:
            logger.error(f"Failed to add song: {e}")
            return False

    def song_exists(self, title: str, artist: str) -> bool:
        """
        Check if a song already exists in the database.

        Args:
            title: Song title
            artist: Artist name

        Returns:
            True if song exists, False otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 1 FROM songs
            WHERE title = ? AND artist = ?
            LIMIT 1
        """, (title, artist))

        return cursor.fetchone() is not None

    def get_song(self, title: str, artist: str) -> Optional[Dict]:
        """
        Retrieve a song from the database.

        Args:
            title: Song title
            artist: Artist name

        Returns:
            Dictionary with song data, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM songs
            WHERE title = ? AND artist = ?
        """, (title, artist))

        row = cursor.fetchone()

        if row:
            return dict(row)

        return None

    def get_song_count(self) -> int:
        """
        Get total number of songs in database.

        Returns:
            Total song count
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM songs")
        return cursor.fetchone()[0]

    def get_stats(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics:
            - total_songs: Total number of songs
            - unique_artists: Number of unique artists
            - avg_lyric_length: Average lyric length
            - by_source: Count by lyrics source
            - by_decade: Count by release decade
        """
        cursor = self.conn.cursor()

        stats = {}

        # Total songs
        cursor.execute("SELECT COUNT(*) FROM songs")
        stats['total_songs'] = cursor.fetchone()[0]

        # Unique artists
        cursor.execute("SELECT COUNT(DISTINCT artist) FROM songs")
        stats['unique_artists'] = cursor.fetchone()[0]

        # Average lyric length
        cursor.execute("SELECT AVG(LENGTH(lyrics)) FROM songs")
        stats['avg_lyric_length'] = cursor.fetchone()[0] or 0

        # By lyrics source
        cursor.execute("""
            SELECT lyrics_source, COUNT(*) as count
            FROM songs
            GROUP BY lyrics_source
        """)
        stats['by_source'] = {row[0] or 'unknown': row[1] for row in cursor.fetchall()}

        # By decade
        cursor.execute("""
            SELECT (release_year / 10) * 10 as decade, COUNT(*) as count
            FROM songs
            WHERE release_year IS NOT NULL
            GROUP BY decade
            ORDER BY decade
        """)
        stats['by_decade'] = {int(row[0]): row[1] for row in cursor.fetchall()}

        return stats

    def export_to_csv(
        self,
        output_path: str,
        include_metadata: bool = False
    ) -> int:
        """
        Export songs to CSV file for existing pipeline integration.

        Args:
            output_path: Path to output CSV file
            include_metadata: If True, include all metadata columns.
                             If False, only export (song_id, title, artist, lyrics)

        Returns:
            Number of songs exported
        """
        if include_metadata:
            query = "SELECT * FROM songs"
        else:
            # Export only core columns needed by existing pipeline
            query = "SELECT song_id, title, artist, lyrics FROM songs"

        cursor = self.conn.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()

        if not rows:
            logger.warning("No songs to export")
            return 0

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])

        # Create parent directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)

        logger.info(f"Exported {len(df)} songs to {output_path}")
        logger.info(f"File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

        return len(df)

    def get_all_songs(self) -> List[Dict]:
        """
        Retrieve all songs from database.

        Returns:
            List of song dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM songs ORDER BY artist, title")

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
        logger.debug("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
