# Research-Assist

A lightweight Gradio-based interface for searching, downloading, indexing, and conversing with academic papers using
retrieval-augmented generation (RAG) and speech services.

## Features

- Search for papers by topic or arXiv ID
- Download and store PDFs locally
- Ingest and index papers for chat-based question answering
- Multiple chat modes: **Paper**, **Author**, and **Analyst**
- Speech-to-text with Whisper and text-to-speech with Piper
- Simple UI built with Gradio for demo and experimentation

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <repo-url> && cd ResearchAssist
   ```

2. **Create a Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python researchassist_app.py
   ```

   The Gradio UI will start on `http://0.0.0.0:7860`. Use the **Paper Selection** tab to search or load existing PDFs, then
   switch to the **Chat Interface** tab to interact with the indexed document.

## Requirements

See `requirements.txt` for the full list of Python packages used by the project. At a minimum, the following are
required:

- `gradio` for the web interface
- `langchain` and related packages for RAG & LLM orchestration
- `chromadb` / `sentence-transformers` for vector stores and embeddings
- `pypdf` / `pymupdf` for PDF handling
- `arxiv` for paper search/download
- `openai-whisper`, `torch`, and `torchaudio` for speech recognition
- `piper-tts` for text-to-speech synthesis

## Directory Layout

```
ResearchAssist/
├── Agent/               # RAG agent code
├── Services/            # Paper management, STT and TTS service modules
├── models/              # Pre‑downloaded voice models for TTS
├── researchassist_app.py# Gradio application entrypoint
├── requirements.txt     # Python dependencies
└── README.md            # This documentation
```

## License

MIT License – see `LICENSE` if present.
