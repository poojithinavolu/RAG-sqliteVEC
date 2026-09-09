"""Minimal RAG chatbot: sqlite-vec vector DB + fastembed embeddings + Claude.

Run:  uvicorn app:app --reload
Then open http://127.0.0.1:8000
"""

import io
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import sqlite_vec
from groq import Groq

load_dotenv()  # read GROQ_API_KEY from a .env file if present
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastembed import TextEmbedding
from pydantic import BaseModel

# --- config ---------------------------------------------------------------
HERE = Path(__file__).parent
DB_PATH = HERE / "rag.db"
MODEL = "openai/gpt-oss-120b"   # any Groq chat model, e.g. "openai/gpt-oss-20b", "qwen/qwen3.8-27b"
EMBED_DIM = 384              # BAAI/bge-small-en-v1.5
TOP_K = 5                    # chunks retrieved per question
CHUNK_SIZE = 1000            # characters
CHUNK_OVERLAP = 200

# --- one-time setup -----------------------------------------------------------
print("Loading embedding model (first run downloads ~90MB)...")
embedder = TextEmbedding()   # BAAI/bge-small-en-v1.5, 384-dim, runs locally
groq = Groq()                # needs GROQ_API_KEY in the environment
app = FastAPI(title="RAG chatbot")


def connect() -> sqlite3.Connection:
    """Open the DB with sqlite-vec loaded and the chunks table ready."""
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0("
        f"  embedding float[{EMBED_DIM}], +source text, +text text)"
    )
    return db


def embed(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in embedder.embed(texts)]


def chunk(text: str) -> list[str]:
    step = CHUNK_SIZE - CHUNK_OVERLAP
    pieces = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step)]
    return [p.strip() for p in pieces if p.strip()]


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


# --- routes -----------------------------------------------------------------
@app.get("/")
def home():
    return FileResponse(HERE / "index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    text = extract_text(file.filename, await file.read())
    chunks = chunk(text)
    if not chunks:
        raise HTTPException(400, "No readable text found in that file.")

    db = connect()
    with db:
        for piece, vector in zip(chunks, embed(chunks)):
            db.execute(
                "INSERT INTO chunks(embedding, source, text) VALUES (?, ?, ?)",
                (sqlite_vec.serialize_float32(vector), file.filename, piece),
            )
    db.close()
    return {"filename": file.filename, "chunks": len(chunks)}


class ChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant", "content": "..."}]


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.messages[-1]["content"]
    query_vec = embed([question])[0]

    db = connect()
    hits = db.execute(
        "SELECT source, text FROM chunks "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (sqlite_vec.serialize_float32(query_vec), TOP_K),
    ).fetchall()
    db.close()

    if not hits:
        return {"reply": "No documents uploaded yet. Add a file first.", "sources": []}

    context = "\n\n".join(f"[{source}]\n{text}" for source, text in hits)
    system = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}"
    )
    resp = groq.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "system", "content": system}, *req.messages],
    )
    reply = resp.choices[0].message.content
    return {"reply": reply, "sources": sorted({source for source, _ in hits})}
