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

1. Enter a mood sentence.
   - HarmonyAI projects text into a mood space by scanning for explicit emotion keywords (e.g., *sad, anxious, motivated*).  
   - It **does not** infer tone of voice, sarcasm, slang, or prompts unrelated to how you feel. Clear emotion words produce the best playlists.
   - Recommended placeholder phrasing: “Describe your current mood using clear emotional language (e.g., ‘tired but hopeful’, ‘calm and focused’).”
   - Valid prompts include concise emotion descriptors such as *happy, sad, calm, tired, energetic, anxious, hopeful, overwhelmed, peaceful,* or *angry*. These keywords give the analyzer something concrete to match.
2. Submit the input.
3. View song recommendations ranked by emotional similarity.

## Limitations

Manual and qualitative testing revealed a limitation in the current sentiment analysis approach when processing highly charged, colloquial, or all-caps user input. For example, the prompt “HOW COULD SHE SMASH MY HEART LIKE THAT AFTER ALL THAT I DID FOR HER?!” was incorrectly mapped to a mood profile dominated by positive attributes such as happy, calm, and energetic. This behavior highlights a known limitation of rule-based sentiment models such as VADER, which can struggle to interpret slang, emotional intensity, tone of speech, or prompts that avoid explicit emotion words. As a result, certain emotionally intense prompts may be inaccurately projected into the three-axis mood wheel representation.

Current analyzer constraints:
- Relies on explicit emotion keywords (sad, hopeful, anxious, calm); prompts without them may be misread.
- Does not capture tone/sarcasm, shouty all-caps phrasing, or figurative language.
- Slang-heavy or off-topic prompts (e.g., narratives without feelings) generally return low-quality recommendations.

## Future Work

- Integrate neural sentence-embedding models (e.g., BERT or Sentence Transformers) so the system captures nuanced, high-arousal user input. This upgrade is critical to address the system’s observed weakness with high-arousal and colloquial text prompts.
