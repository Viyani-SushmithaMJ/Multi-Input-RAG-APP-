
## 🎧 Multi-Modal Q&A with LangChain + Whisper + FAISS

An interactive Streamlit app that answers questions from YouTube videos, uploaded documents, and audio using OpenAI LLMs. It transcribes, embeds, and retrieves relevant chunks using Whisper and FAISS for accurate, source-based Q&A.

setup/Run steps:

1.pip install -r requirements.txt

2.Set OpenAI API Key

3.streamlit run app.py


Features
--------
- Ingest from multiple sources: YouTube URL, file uploads (PDF/TXT/DOCX/etc.), and audio.
- Transcription with Whisper (API or local).
- Vector search with FAISS for fast, semantic retrieval.
- LLM answers grounded in retrieved chunks (source-aware).
- Streamlit UI for simple, interactive use.


Quick Start
-----------
1) Install dependencies
   pip install -r requirements.txt

2) Run the app
   streamlit run app.py

3) In the browser UI:
   - Paste a YouTube URL OR upload files/audio
   - Click “Ingest” (if provided) or follow the prompts
   - Ask questions and see answers with cited chunks



