# Egyptian Artifacts Recognition 🔥

An AI-powered image similarity system for recognizing Egyptian artifacts.

## Features
- Image similarity using EfficientNet
- FAISS fast search
- Streamlit UI
- Wikipedia integration

## Run

```bash
pip install -r requirements.txt
python src/2_extract_features.py
python src/4_build_index.py
streamlit run src/app.py