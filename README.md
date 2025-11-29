# HarmonyAI

HarmonyAI turns mood descriptions into playlist recommendations using a simple mood wheel model.

## What's here
- `demo_harmony.py` – CLI demo of the full pipeline
- `streamlit_app.py` – Streamlit UI (hero, emotion charts, playlist cards)
- `mood_model.py` – keyword analyzer, projection logic, similarity ranking

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the CLI demo
```bash
python demo_harmony.py
```

## Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

## Run tests
```bash
python -m pytest
```
