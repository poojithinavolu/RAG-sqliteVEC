# RAG Chatbot

A minimal retrieval-augmented chatbot. Upload a document, ask questions about it.

- **Vector DB:** [sqlite-vec](https://github.com/asg017/sqlite-vec) — a single-file `vec0` virtual table in `rag.db`
- **Embeddings:** `BAAI/bge-small-en-v1.5` via [fastembed](https://github.com/qdrant/fastembed), runs locally, no API key
- **LLM:** Groq (`llama-3.3-70b-versatile` by default — edit `MODEL` in `app.py`)
- **UI:** one static `index.html`, no build step

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
echo GROQ_API_KEY=gsk_... > .env   # or set it in your environment
```

## Run

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 — upload a `.txt`, `.md`, `.pdf`, `.csv`, `.json`, or `.log` file, then chat.

## How it works

1. **Upload** — the file's text is split into ~1000-char overlapping chunks, each embedded and stored in `rag.db`.
2. **Chat** — your question is embedded, the 5 nearest chunks are pulled from sqlite-vec with a KNN query, and they're handed to Groq as context.

The database persists between runs. Delete `rag.db` to start fresh.
