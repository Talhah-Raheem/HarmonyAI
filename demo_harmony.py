"""
Simple CLI demo for HarmonyAI showing how to run the mood-to-music pipeline end to end.
Uses a tiny hard-coded catalog with placeholder mood vectors; real systems will swap in
lyrics-derived vectors and richer metadata.
"""

from typing import List

import numpy as np
import pandas as pd

from mood_model import HarmonyMoodModel, MoodVector


def main() -> None:
    # Configure the axes that define the mood space for this demo recommender.
    mood_axes: List[str] = ["valence", "energy", "tension"]
    model = HarmonyMoodModel(mood_axes=mood_axes)

    # Tiny in-memory catalog standing in for a real song database with NLP-extracted vectors.
    songs = pd.DataFrame(
        [
            {
                "title": "Sunrise Optimism",
                "artist": "Aurora Sky",
                "mood_vector": np.array([0.9, 0.6, -0.4]),
            },
            {
                "title": "Midnight Reflections",
                "artist": "Lunar Echo",
                "mood_vector": np.array([-0.6, -0.4, 0.5]),
            },
            {
                "title": "Raging Storm",
                "artist": "Thunder Pulse",
                "mood_vector": np.array([-0.8, 0.9, 0.9]),
            },
            {
                "title": "Ocean Breeze",
                "artist": "Calm Current",
                "mood_vector": np.array([0.7, -0.5, -0.8]),
            },
        ]
    )

    user_text = input("Describe your current mood: ")

    # Run the HarmonyMoodModel pipeline: analyze text, project to axes, rank songs, explain picks.
    emotion_scores = model.analyze_text(user_text)
    user_mood = model.project_to_mood_wheel(emotion_scores)
    ranked = model.score_songs_for_mood(user_mood, songs, top_k=3)

    print("\nTop song matches for your mood:\n")
    for _, row in ranked.iterrows():
        explanation = model.explain_song_match(user_mood, row["mood_vector"])
        print(f"- {row['title']} by {row['artist']} (similarity={row['similarity']:.3f})")
        print(f"  Why: {explanation}")


if __name__ == "__main__":
    main()
