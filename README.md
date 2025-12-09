# HarmonyAI 🎵

HarmonyAI is a mood-based music recommendation system.  
Users enter a short sentence describing how they feel (e.g., *“I feel sad but hopeful”*), and the system analyzes the emotional content of the text to recommend songs that best match the detected mood.

The project runs entirely locally and does **not require any external APIs**.

---

## Requirements
- Python 3.9+
- Windows 10/11 **or** macOS
- PowerShell (Windows) or Terminal (macOS)

---

## Setup & Run

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

python -m scripts.augment_data
python -m scripts.build_index
python -m scripts.build_embeddings

streamlit run streamlit_app.py
```
### MacOS (Terminal)
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m scripts.augment_data
python -m scripts.build_index
python -m scripts.build_embeddings

streamlit run streamlit_app.py
```

## Expanding the Dataset (Optional)

HarmonyAI is designed to run entirely offline using the included song dataset.  
However, the system can be extended to support larger catalogs by integrating the Genius Lyrics API.

To expand the dataset:
1. Obtain a Genius API key from https://genius.com/api-clients
2. Create a `.env` file in the project root
3. Add your API key: GENIUS_API_TOKEN=your_api_key_here
4. Modify or extend the data ingestion scripts to fetch additional songs and lyrics. This step is **optional** and not required to run the current version of the project.


### Using the App

Enter a mood sentence

Submit the input

View song recommendations ranked by emotional similarity
