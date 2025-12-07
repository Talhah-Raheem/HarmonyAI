# Run the full pipeline and launch Streamlit
$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Path $MyInvocation.MyCommand.Path -Parent)

python -m scripts.augment_data
python -m scripts.build_index
python -m scripts.build_embeddings

python -m streamlit run streamlit_app.py
