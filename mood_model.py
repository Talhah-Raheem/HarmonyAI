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
        # Expanded to 7 categories for a well-rounded analysis
        emotion_keywords: Dict[str, List[str]] = {
            "happy": [
                "happy", "joy", "joyful", "hopeful", "positive", "cheerful", 
                "glad", "delighted", "content", "pleased", "good", "great"
            ],
            "sad": [
                "sad", "down", "depressed", "lonely", "melancholy", "blue", 
                "gloomy", "sorrowful", "heartbroken", "disappointed", "miserable"
            ],
            "angry": [
                "angry", "mad", "furious", "enraged", "outraged", "hostile", 
                "resentful", "hateful", "bitter", "annoyed", "frustrated"
            ],
            "calm": [
                "calm", "relaxed", "peaceful", "chill", "serene", "tranquil", 
                "mellow", "composed", "zen", "quiet", "still", "meditate"
            ],
            # NEW: Handles "motivated", "pumped", "work" natively
            "energetic": [
                "excited", "energetic", "motivated", "pumped", "hyped", 
                "driven", "active", "dynamic", "power", "ambitious", 
                "productive", "fast", "awake", "dance", "workout"
            ],
            # NEW: Handles "stress" and "nerves" separately from Anger
            "anxious": [
                "anxious", "nervous", "worried", "stressed", "tense", 
                "uneasy", "scared", "fearful", "panicked", "restless", 
                "overwhelmed", "pressure"
            ],
            # NEW: The home for "tired" (Low Energy, Neutral/Negative)
            "tired": [
                "tired", "exhausted", "drained", "sleepy", "fatigue", 
                "weary", "burnout", "beat", "worn out", "lazy", 
                "sleep", "bed", "napping"
            ]
        }

        # Negation patterns (Keep these!)
        negation_patterns = [
            "not ", "don't ", "dont ", "no ", "never ", "isn't ", "isnt ", 
            "aren't ", "arent ", "wasn't ", "wasnt ", "won't ", "wont ", 
            "can't ", "cant ", "wouldn't ", "wouldnt "
        ]

        text_lower = text.lower()
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        raw_counts: Dict[str, float] = {}

        for emotion, keywords in emotion_keywords.items():
            count = 0.0
            for keyword in keywords:
                keyword_positions = []
                start = 0
                while True:
                    pos = text_lower.find(keyword, start)
                    if pos == -1:
                        break
                    keyword_positions.append(pos)
                    start = pos + 1

                for pos in keyword_positions:
                    context_start = max(0, pos - 20)
                    context = text_lower[context_start:pos]
                    is_negated = any(neg in context for neg in negation_patterns)

                    if is_negated:
                        count -= 0.5
                    else:
                        count += 1.0

            raw_counts[emotion] = max(0.0, float(count))

        # (Removed the old "motivation_keywords" logic block here)

        pos = max(sentiment.get("pos", 0.0), 0.0)
        neg = max(sentiment.get("neg", 0.0), 0.0)
        neu = max(sentiment.get("neu", 0.0), 0.0)
        compound = sentiment.get("compound", 0.0)

        # Updated scoring logic for 7 categories
        scores: Dict[str, float] = {}
        scores["happy"]     = raw_counts["happy"] + max(compound, 0.0) + 0.5 * pos
        scores["calm"]      = raw_counts["calm"] + 0.6 * pos + 0.4 * neu
        scores["sad"]       = raw_counts["sad"] + max(-compound, 0.0) + 0.5 * neg
        scores["angry"]     = raw_counts["angry"] + 0.7 * neg + 0.3 * max(-compound, 0.0)
        scores["energetic"] = raw_counts["energetic"] + 0.5 * pos + 0.3 * max(compound, 0.0)
        scores["anxious"]   = raw_counts["anxious"] + 0.6 * neg + 0.4 * max(-compound, 0.0)
        scores["tired"]     = raw_counts["tired"] + 0.5 * neg + 0.5 * neu

        total = sum(scores.values())
        if total == 0:
            uniform_score = 1.0 / len(emotion_keywords)
            return {emotion: uniform_score for emotion in emotion_keywords}

        return {emotion: value / total for emotion, value in scores.items()}

    # Projects the emotion scores onto the configured mood wheel to produce a MoodVector.
    def project_to_mood_wheel(self, emotion_scores: Dict[str, float]) -> MoodVector:
        # Map 7 discrete emotions to the 3 continuous axes
        emotion_to_axes: Dict[str, Dict[str, float]] = {
            "happy":     {"valence": 1.0,  "energy": 0.6,  "tension": -0.2},
            "sad":       {"valence": -1.0, "energy": -0.4, "tension": 0.2},
            "angry":     {"valence": -0.8, "energy": 0.8,  "tension": 1.0},
            "calm":      {"valence": 0.5,  "energy": -0.8, "tension": -0.9},
            "energetic": {"valence": 0.8,  "energy": 1.0,  "tension": 0.1},
            "anxious":   {"valence": -0.5, "energy": 0.5,  "tension": 1.0},
            "tired":     {"valence": -0.2, "energy": -1.0, "tension": -0.3}
        }

        wheel_values = np.zeros(len(self.mood_axes), dtype=float)
        for emotion, score in emotion_scores.items():
            axis_weights = emotion_to_axes.get(emotion, {})
            for index, axis in enumerate(self.mood_axes):
                weight = axis_weights.get(axis, 0.0)
                wheel_values[index] += score * weight

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
