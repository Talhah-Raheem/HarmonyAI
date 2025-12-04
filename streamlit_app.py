"""
Modern Streamlit interface for HarmonyAI, turning mood prompts into curated playlists.
"""

from __future__ import annotations

from typing import Dict, List

import html
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from mood_model import HarmonyMoodModel, MoodVector


@st.cache_data
def build_demo_catalog() -> pd.DataFrame:
    """Returns a tiny in-memory catalog with placeholder mood vectors."""
    return pd.DataFrame(
        [
            {
                "title": "Sunrise Optimism",
                "artist": "Aurora Sky",
                "mood_vector": np.array([0.9, 0.6, -0.4]),
                "genre": "Indie Pop",
                "accent": "#7c3aed",
            },
            {
                "title": "Midnight Reflections",
                "artist": "Lunar Echo",
                "mood_vector": np.array([-0.6, -0.4, 0.5]),
                "genre": "Lo-fi",
                "accent": "#0ea5e9",
            },
            {
                "title": "Raging Storm",
                "artist": "Thunder Pulse",
                "mood_vector": np.array([-0.8, 0.9, 0.9]),
                "genre": "Hard Rock",
                "accent": "#f97316",
            },
            {
                "title": "Ocean Breeze",
                "artist": "Calm Current",
                "mood_vector": np.array([0.7, -0.5, -0.8]),
                "genre": "Acoustic",
                "accent": "#10b981",
            },
        ]
    )


@st.cache_data
def load_catalog() -> pd.DataFrame:
    """Load processed song catalog and compute mood vectors from lyrics."""
    import pandas as pd
    from mood_model import HarmonyMoodModel
    import numpy as np
    import random

    # Load cleaned dataset
    df = pd.read_csv('data/processed/songs_clean.csv')

    # Initialize mood model
    mood_model = HarmonyMoodModel(['valence', 'energy', 'tension'])

    # Compute mood vectors from lyrics
    def compute_mood_vector(lyrics: str) -> np.ndarray:
        try:
            emotion_scores = mood_model.analyze_text(lyrics)
            mood_vec = mood_model.project_to_mood_wheel(emotion_scores)
            return mood_vec.values
        except:
            # Fallback to neutral mood if analysis fails
            return np.array([0.0, 0.0, 0.0])

    df['mood_vector'] = df['lyrics'].apply(compute_mood_vector)

    # Add genre (placeholder - could be enhanced later)
    genres = ['Pop', 'Rock', 'Indie', 'Alternative', 'Electronic', 'Folk', 'Hip-Hop', 'R&B']
    df['genre'] = [random.choice(genres) for _ in range(len(df))]

    # Add accent color based on mood
    def mood_to_color(mood_vec: np.ndarray) -> str:
        valence, energy, tension = mood_vec
        if valence > 0.3 and energy > 0.3:
            return "#7c3aed"  # Purple - upbeat
        elif valence < -0.3 and energy < 0:
            return "#0ea5e9"  # Blue - sad
        elif tension > 0.5 and energy > 0.5:
            return "#f97316"  # Orange - intense
        else:
            return "#10b981"  # Green - calm

    df['accent'] = df['mood_vector'].apply(mood_to_color)

    return df


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
            .main {
                background: radial-gradient(circle at 20% 20%, #10121a, #05070d 55%);
                color: #f3f4f6;
                font-family: 'Space Grotesk', sans-serif;
                padding-bottom: 4rem;
            }
            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #080a12, #05070d);
                color: #c3c7d1;
                font-family: 'Space Grotesk', sans-serif;
                border-right: 1px solid rgba(255, 255, 255, 0.04);
                padding-top: 1.5rem;
            }
            div[data-testid="stSidebar"] .sidebar-shell {
                padding: 0 1rem 2rem;
            }
            .sidebar-card {
                background: rgba(8, 9, 15, 0.8);
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                padding: 1rem 1.25rem;
                margin-bottom: 1rem;
            }
            .sidebar-card h4 {
                margin: 0 0 0.4rem;
                font-size: 1rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                color: #f3f4f6;
            }
            .sidebar-card p {
                margin: 0 0 0.6rem;
                color: #aab3c1;
                font-size: 0.9rem;
            }
            .hero {
                padding: 1.5rem 2rem;
                border-radius: 20px;
                background: linear-gradient(120deg, #1f1b55, #2e265f 35%, #3f1b5c 100%);
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 25px 60px rgba(6, 7, 12, 0.9);
                margin-bottom: 2rem;
            }
            .hero h1 {
                font-size: 2.75rem;
                margin-bottom: 0.3rem;
            }
            .hero p {
                margin: 0.4rem 0;
                color: rgba(255, 255, 255, 0.82);
            }
            .panel {
                background: rgba(11, 14, 22, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 18px;
                padding: 1.5rem;
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
            }
            .panel textarea, .panel input, .panel select {
                font-family: 'Space Grotesk', sans-serif;
            }
            .emotion-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 1rem;
            }
            @media (max-width: 768px) {
                .hero {
                    padding: 1.25rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(mood_axes: List[str]) -> int:
    st.sidebar.markdown('<div class="sidebar-shell">', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <h4>Harmony Controls</h4>
            <p>Set playlist size to match the level of variety you want.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_k = st.sidebar.slider("Playlist size", min_value=1, max_value=10, value=5)

    axis_descriptions = {
        "valence": "Sad ↔ Happy",
        "energy": "Calm ↔ Energetic",
        "tension": "Relaxed ↔ Intense",
    }
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <h4>Mood Dimensions</h4>
            <p>How we analyze your emotional state.</p>
        """,
        unsafe_allow_html=True,
    )
    for axis in mood_axes:
        st.sidebar.write(f"**{axis.title()}** · {axis_descriptions.get(axis, 'Custom axis')}")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <h4>About</h4>
            <p>HarmonyAI team · Mood vectors + playlist explorer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    return top_k


def render_hero_section() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>HarmonyAI</h1>
            <p>Describe your emotional state, then let HarmonyAI translate it into a curated listening session. Built with a custom mood wheel, projection logic, and transparent explanations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mood_prompt_form() -> str | None:
    with st.form("mood_form"):
        st.subheader("Tell HarmonyAI How You're Feeling")
        user_text = st.text_area(
            "Mood prompt",
            value="I'm tired from school but trying to stay motivated for work tonight.",
            height=150,
            placeholder="E.g., I'm drained but trying to stay positive about what's ahead.",
        )
        submitted = st.form_submit_button("Generate Playlist", use_container_width=True)
    if not submitted:
        return None
    if not user_text.strip():
        st.warning("Please enter a few words describing your mood.")
        return ""
    return user_text


def display_emotion_breakdown(emotion_scores: Dict[str, float]) -> None:
    st.subheader("Emotion Pulse")
    sorted_scores = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
    cards = []
    for emotion, score in sorted_scores:
        percent = min(score, 1.0) * 100
        cards.append(
            f"""
            <div class="emotion-card">
                <p class="emotion-label">{emotion}</p>
                <p class="emotion-score">{score:.2f}</p>
                <div class="emotion-bar">
                    <div class="emotion-bar-fill" style="width:{percent}%"></div>
                </div>
            </div>
            """
        )

    cards_html = "".join(cards)
    grid_html = f"""
    <style>
        .emotion-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
        }}
        .emotion-card {{
            background: rgba(11, 14, 22, 0.92);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 1.3rem 1.4rem;
            font-family: 'Space Grotesk', sans-serif;
        }}
        .emotion-label {{
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin: 0;
            font-size: 0.85rem;
        }}
        .emotion-score {{
            font-size: 2rem;
            font-weight: 600;
            margin: 0.4rem 0 0;
            color: #f8fafc;
        }}
        .emotion-bar {{
            margin-top: 0.5rem;
            height: 8px;
            border-radius: 999px;
            background: rgba(148,163,184,0.18);
            overflow: hidden;
        }}
        .emotion-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg,#60a5fa,#a855f7);
        }}
    </style>
    <div class="emotion-grid">
        {cards_html}
    </div>
    """
    rows = math.ceil(len(sorted_scores) / 2) or 1
    height = max(180, rows * 160)
    components.html(grid_html, height=height, scrolling=False)


def display_mood_radar(user_mood: MoodVector) -> None:
    values = user_mood.values.tolist()
    axes = user_mood.axes
    values.append(values[0])
    axes_cycle = axes + [axes[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=axes_cycle,
                fill="toself",
                line=dict(color="#a855f7"),
                name="Mood profile",
            )
        ]
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(15,23,42,0.4)",
            radialaxis=dict(visible=True, range=[-1, 1], showline=False, showticklabels=False),
            angularaxis=dict(showline=False, tickfont=dict(color="#f3f4f6")),
        ),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f4f6"),
    )
    st.subheader("Mood Wheel Projection")
    st.plotly_chart(fig, use_container_width=True)


def build_icon_svg(color: str) -> str:
    return f"""
    <svg viewBox="0 0 80 80" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="{color}" stop-opacity="0.35"/>
                <stop offset="100%" stop-color="{color}" stop-opacity="0.1"/>
            </linearGradient>
        </defs>
        <rect x="8" y="8" width="64" height="64" rx="20" fill="url(#grad)" stroke="{color}" stroke-opacity="0.6"/>
        <path d="M36 25L52 34L36 43V25Z" fill="{color}" fill-opacity="0.7"/>
        <circle cx="30" cy="49" r="4" fill="{color}" fill-opacity="0.5"/>
        <circle cx="46" cy="49" r="4" fill="{color}" fill-opacity="0.8"/>
    </svg>
    """


def display_recommendations(
    ranked: pd.DataFrame,
    user_mood: MoodVector,
    model: HarmonyMoodModel,
) -> None:
    st.subheader("Curated Picks")
    if ranked.empty:
        st.info("No songs available yet. Bring in datasets from the data team to populate the catalog.")
        return

    card_markup: List[str] = []
    for _, row in ranked.iterrows():
        similarity = row["similarity"]
        explanation = model.explain_song_match(user_mood, row["mood_vector"])
        accent = row.get("accent", "#6366f1")
        icon_svg = build_icon_svg(accent)
        title = html.escape(str(row["title"]))
        artist = html.escape(str(row["artist"]))
        genre = html.escape(str(row.get("genre", "Unknown Genre")))

        card_markup.append(
            f"""
            <article class="playlist-card" style="--accent:{accent};">
                <div class="playlist-cover" style="background: radial-gradient(circle at 30% 20%, {accent} 0%, rgba(0,0,0,0) 60%), rgba(8,10,18,0.95);">
                    <div class="playlist-icon">{icon_svg}</div>
                </div>
                <div class="playlist-body">
                    <div class="playlist-topline">
                        <div>
                            <p class="playlist-label">Artist</p>
                            <h4>{artist}</h4>
                        </div>
                        <div class="playlist-score">
                            <span>Similarity</span>
                            <strong>{similarity:.3f}</strong>
                        </div>
                    </div>
                    <p class="playlist-title">{title}</p>
                    <p class="playlist-meta">{genre}</p>
                    <p class="playlist-copy">{explanation}</p>
                </div>
            </article>
            """
        )

    cards_html = "".join(card_markup)
    playlist_html = f"""
    <style>
        .playlist-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
        }}
        .playlist-card {{
            background: rgba(5, 6, 12, 0.98);
            border-radius: 24px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.04);
            box-shadow: 0 10px 24px rgba(3, 4, 7, 0.28);
            display: flex;
            flex-direction: column;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }}
        .playlist-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 16px 32px rgba(3, 4, 7, 0.35);
        }}
        .playlist-cover {{
            position: relative;
            padding: 1.5rem;
            min-height: 150px;
        }}
        .playlist-icon {{
            width: 70px;
            height: 70px;
        }}
        .playlist-icon svg {{
            width: 70px;
            height: 70px;
        }}
        .playlist-body {{
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            color: #e2e8f0;
        }}
        .playlist-topline {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        .playlist-topline h4 {{
            color: #f8fafc;
        }}
        .playlist-label {{
            margin: 0;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.75rem;
            color: #94a3b8;
        }}
        .playlist-title {{
            font-size: 1.3rem;
            font-weight: 600;
            margin: 0.2rem 0 0;
            color: #f8fafc;
        }}
        .playlist-meta {{
            margin: 0;
            color: #9ca3af;
            font-size: 0.95rem;
        }}
        .playlist-copy {{
            margin-top: 0.5rem;
            color: #d7dce8;
            line-height: 1.6;
        }}
    </style>
    <section class="playlist-grid">
        {cards_html}
    </section>
    """
    rows = math.ceil(len(ranked) / 2) or 1
    height = max(380, rows * 320)
    components.html(playlist_html, height=height, scrolling=True)


def main() -> None:
    st.set_page_config(page_title="HarmonyAI Demo", layout="wide")
    inject_custom_css()
    render_hero_section()

    mood_axes = ["valence", "energy", "tension"]
    model = HarmonyMoodModel(mood_axes=mood_axes)
    songs = load_catalog()  # Load real songs instead of demo
    top_k = render_sidebar(mood_axes)

    mood_prompt = mood_prompt_form()
    if mood_prompt is None:
        return
    if mood_prompt == "":
        return

    emotion_scores = model.analyze_text(mood_prompt)
    user_mood = model.project_to_mood_wheel(emotion_scores)
    ranked = model.score_songs_for_mood(user_mood, songs, top_k=top_k)

    info_col, viz_col = st.columns([1.4, 1])
    with info_col:
        display_emotion_breakdown(emotion_scores)
    with viz_col:
        display_mood_radar(user_mood)

    st.markdown("---")
    display_recommendations(ranked, user_mood, model)


if __name__ == "__main__":
    main()
