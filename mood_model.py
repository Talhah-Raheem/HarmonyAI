"""
Core interfaces for mood-based music recommendations.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Represents a point in the mood space using predefined axes.
@dataclass
class MoodVector:
    values: np.ndarray
    axes: List[str]


# Defines the primary mood model responsible for analyzing user mood and ranking songs.
class HarmonyMoodModel:
    # Initializes the model with the axes that define the mood space.
    def __init__(self, mood_axes: List[str]) -> None:
        self.mood_axes = mood_axes
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    # Analyzes text with a lightweight rule-based detector to seed more advanced models later.
    def analyze_text(self, text: str) -> Dict[str, float]:
        # Combine VADER sentiment with negation-aware keyword cues; remains offline and fast while
        # capturing intensity better than raw substring counts alone.
        emotion_keywords: Dict[str, List[str]] = {
            "happy": [
                "happy",
                "joy",
                "joyful",
                "excited",
                "hopeful",
                "positive",
                "uplifted",
                "cheerful",
                "glad",
                "delighted",
                "content",
                "pleased",
            ],
            "sad": [
                "sad",
                "down",
                "depressed",
                "lonely",
                "melancholy",
                "blue",
                "gloomy",
                "sorrowful",
                "heartbroken",
                "disappointed",
                "miserable",
            ],
            "angry": [
                "angry",
                "mad",
                "frustrated",
                "upset",
                "irritated",
                "furious",
                "annoyed",
                "enraged",
                "outraged",
                "hostile",
                "resentful",
            ],
            "calm": [
                "calm",
                "relaxed",
                "peaceful",
                "chill",
                "serene",
                "tranquil",
                "mellow",
                "composed",
                "centered",
                "balanced",
                "still",
            ],
        }

        # Additional keyword patterns for motivation/energy states
        motivation_keywords = [
            "motivated",
            "energized",
            "driven",
            "focused",
            "determined",
            "ambitious",
            "productive",
            "work",
            "accomplish",
            "achieve",
            "get things done",
            "push through",
            "power through",
        ]

        # Negation patterns to detect reversed sentiment
        negation_patterns = [
            "not ",
            "don't ",
            "dont ",
            "no ",
            "never ",
            "isn't ",
            "isnt ",
            "aren't ",
            "arent ",
            "wasn't ",
            "wasnt ",
            "won't ",
            "wont ",
            "can't ",
            "cant ",
            "wouldn't ",
            "wouldnt ",
        ]

        text_lower = text.lower()
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        raw_counts: Dict[str, float] = {}

        for emotion, keywords in emotion_keywords.items():
            count = 0.0
            for keyword in keywords:
                # Find all occurrences of this keyword
                keyword_positions = []
                start = 0
                while True:
                    pos = text_lower.find(keyword, start)
                    if pos == -1:
                        break
                    keyword_positions.append(pos)
                    start = pos + 1

                # Check each occurrence for negation
                for pos in keyword_positions:
                    # Look for negation in the 20 characters before the keyword
                    context_start = max(0, pos - 20)
                    context = text_lower[context_start:pos]

                    is_negated = any(neg in context for neg in negation_patterns)

                    if is_negated:
                        # Keyword is negated - reduce its count or flip to opposite
                        count -= 0.5
                    else:
                        count += 1.0

            raw_counts[emotion] = max(0.0, float(count))  # Don't allow negative counts

        # Check for motivation/energy keywords
        motivation_count = 0
        for keyword in motivation_keywords:
            if keyword in text_lower:
                motivation_count += 1

        # If user wants motivation/energy, boost happy and reduce sad
        if motivation_count > 0:
            raw_counts["happy"] += motivation_count * 0.5
            raw_counts["sad"] = max(0.0, raw_counts["sad"] - motivation_count * 0.3)
            # Boost energy by reducing calm
            raw_counts["calm"] = max(0.0, raw_counts["calm"] - motivation_count * 0.2)

        pos = max(sentiment.get("pos", 0.0), 0.0)
        neg = max(sentiment.get("neg", 0.0), 0.0)
        neu = max(sentiment.get("neu", 0.0), 0.0)
        compound = sentiment.get("compound", 0.0)

        # Blend sentiment polarity with keyword hints to derive per-emotion signals.
        scores: Dict[str, float] = {}
        scores["happy"] = raw_counts["happy"] + max(compound, 0.0) + 0.5 * pos
        scores["calm"] = raw_counts["calm"] + 0.6 * pos + 0.4 * neu
        scores["sad"] = raw_counts["sad"] + max(-compound, 0.0) + 0.5 * neg
        scores["angry"] = raw_counts["angry"] + 0.7 * neg + 0.3 * max(-compound, 0.0)

        total = sum(scores.values())
        if total == 0:
            # If no signals are detected, fall back to uniform tiny scores to avoid zero vectors.
            uniform_score = 1.0 / len(emotion_keywords)
            return {emotion: uniform_score for emotion in emotion_keywords}

        # Normalizing keeps outputs in [0, 1] regardless of text length or sentiment magnitude.
        return {emotion: value / total for emotion, value in scores.items()}

    # Projects the emotion scores onto the configured mood wheel to produce a MoodVector.
    def project_to_mood_wheel(self, emotion_scores: Dict[str, float]) -> MoodVector:
        # The mood wheel maps discrete emotion labels into continuous axes (e.g., valence, energy)
        # so that downstream ranking can compare user mood to song moods in the same space.
        emotion_to_axes: Dict[str, Dict[str, float]] = {
            "happy": {"valence": 1.0, "energy": 0.8, "tension": -0.5},
            "sad": {"valence": -1.0, "energy": -0.5, "tension": 0.4},
            "angry": {"valence": -0.8, "energy": 1.0, "tension": 1.0},
            "calm": {"valence": 0.6, "energy": -0.6, "tension": -0.7},
        }

        # Start with a neutral vector and accumulate weighted contributions for each axis.
        wheel_values = np.zeros(len(self.mood_axes), dtype=float)
        for emotion, score in emotion_scores.items():
            axis_weights = emotion_to_axes.get(emotion, {})
            for index, axis in enumerate(self.mood_axes):
                weight = axis_weights.get(axis, 0.0)
                wheel_values[index] += score * weight

        # Clip keeps the projection bounded within the theoretical limits of the mood wheel,
        # preventing extreme scores from dominating and ensuring consistent downstream usage.
        wheel_values = np.clip(wheel_values, -1.0, 1.0)
        return MoodVector(values=wheel_values, axes=self.mood_axes)

    # Scores each song against the user's mood; later will use cosine similarity between vectors.
    def score_songs_for_mood(
        self,
        user_mood: MoodVector,
        song_moods: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        # Cosine similarity measures the angle between two vectors, giving a scale-invariant
        # score in [-1, 1] that highlights directional alignment of moods instead of magnitude.
        def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0.0 or norm_b == 0.0:
                # Zero-norm vectors cannot produce a meaningful similarity; treat as no match.
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        scored = song_moods.copy()
        scored["similarity"] = scored["mood_vector"].apply(
            lambda song_vec: cosine_similarity(user_mood.values, song_vec)
        )

        # Sorting by similarity ensures the most aligned moods appear first for recommendation.
        scored = scored.sort_values(by="similarity", ascending=False)
        return scored.head(top_k)

    # Provides a natural-language explanation linking user mood to a song's mood vector.
    def explain_song_match(
        self,
        user_mood: MoodVector,
        song_mood: np.ndarray,
    ) -> str:
        # Interpretability builds trust in recommenders by revealing why a song was chosen.
        if len(song_mood) != len(self.mood_axes):
            raise ValueError("Song mood vector length must match the configured mood axes.")

        # Comparing each axis surfaces concrete evidence (e.g., valence, energy) that users
        # can relate to, converting abstract vectors into human-readable reasoning.
        similarities: List[str] = []
        differences: List[str] = []
        for axis, user_value, song_value in zip(user_mood.axes, user_mood.values, song_mood):
            if user_value > 0 and song_value > 0:
                similarities.append(f"both emphasize high {axis}")
            elif user_value < 0 and song_value < 0:
                similarities.append(f"both share low {axis}")
            elif user_value * song_value < 0:
                descriptor = "balances" if abs(song_value) < abs(user_value) else "contrasts"
                differences.append(f"the song {descriptor} your {axis}")

        # Tie the explanation back into the pipeline by anchoring on the strongest axis from the
        # mood projection, then describing how the ranked song relates to that dominant feeling.
        strongest_idx = int(np.argmax(np.abs(user_mood.values)))
        strongest_axis = user_mood.axes[strongest_idx]
        strongest_value = user_mood.values[strongest_idx]
        axis_direction = (
            "high" if strongest_value > 0 else "low" if strongest_value < 0 else "balanced"
        )
        song_axis_value = song_mood[strongest_idx]
        alignment_score = strongest_value * song_axis_value
        if alignment_score > 0:
            axis_alignment = "aligns with"
        elif alignment_score < 0:
            axis_alignment = "contrasts"
        else:
            axis_alignment = "complements"

        explanation = f"This song {axis_alignment} your {axis_direction}-{strongest_axis} mood"
        if similarities:
            explanation += f", with {', '.join(similarities[:2])}"
        if differences:
            connective = "while" if similarities else "and"
            explanation += f", {connective} {', '.join(differences[:2])}"
        return explanation + "."
