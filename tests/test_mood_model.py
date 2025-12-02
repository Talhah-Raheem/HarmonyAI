import sys
from pathlib import Path

import pytest

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mood_model import HarmonyMoodModel  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return HarmonyMoodModel(mood_axes=["valence", "energy", "tension"])


def test_happy_prompt(model):
    scores = model.analyze_text("I feel happy and excited, everything is awesome!")
    assert max(scores, key=scores.get) == "happy"


def test_calm_prompt(model):
    scores = model.analyze_text("Feeling peaceful, calm, and relaxed after yoga.")
    assert max(scores, key=scores.get) == "calm"


def test_angry_prompt(model):
    scores = model.analyze_text("I'm furious, so angry and upset right now!")
    assert max(scores, key=scores.get) == "angry"


def test_sad_prompt(model):
    scores = model.analyze_text("I'm really sad and down; it's a depressing day.")
    assert max(scores, key=scores.get) == "sad"


def test_scores_normalize(model):
    scores = model.analyze_text("Neutral statement without strong emotion.")
    assert pytest.approx(sum(scores.values()), rel=1e-6) == 1.0
