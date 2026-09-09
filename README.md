# RAG Chatbot — a test drive of `sqlite-vec`

## Why this exists

I wanted to find out whether [**sqlite-vec**](https://github.com/asg017/sqlite-vec) is any good as a
vector database for real retrieval work — not read about it, actually build something on it.

So this repo is a small, complete RAG chatbot: upload a document, ask questions, get answers
grounded in that document. Every vector operation — storage, indexing, nearest-neighbour search —
goes through `sqlite-vec` and nothing else. No Chroma, no FAISS, no Pinecone, no pgvector.

**Verdict: it works perfectly fine.** It installed in seconds, the API is plain SQL, the whole
"database" is one file, and every query returned the right chunks. For a project of this size I
wouldn't reach for anything heavier. Details below.

---

## What `sqlite-vec` is

A SQLite extension (single C file, no dependencies) that adds vector search to any SQLite database.
You `pip install sqlite-vec`, load the extension into a normal `sqlite3` connection, and you get a
virtual table type called `vec0` that stores vectors and answers KNN queries with the `MATCH`
operator. That's the entire surface area.

## How this project uses it

**1. Load the extension into a normal connection** ([app.py](app.py)):

```python
db = sqlite3.connect("rag.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)
```

**2. Create one `vec0` table.** The `+` columns are "auxiliary" columns — sqlite-vec stores them
alongside each vector and hands them back on search, so there's no separate metadata table and no
JOIN:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(
    embedding float[384],
    +source   text,
    +text     text
);
```

**3. Insert** — vectors go in as packed float32 bytes via the helper the library ships:

```python
db.execute(
    "INSERT INTO chunks(embedding, source, text) VALUES (?, ?, ?)",
    (sqlite_vec.serialize_float32(vector), filename, chunk_text),
)
```

**4. Search** — one query does the KNN. `MATCH` + `k` is all it takes:

```sql
SELECT source, text
FROM chunks
WHERE embedding MATCH ? AND k = ?
ORDER BY distance;
```

That's the whole vector layer. ~10 lines of Python plus two SQL statements.

---

## How the chatbot works

```
          upload                              chat
 ┌──────────────────────┐          ┌──────────────────────────┐
 │ file → text          │          │ question → embedding      │
 │ → ~1000-char chunks  │          │ → sqlite-vec KNN (top 5)  │
 │ → embed each locally │          │ → chunks + question       │
 │ → INSERT into vec0   │          │   sent to Groq LLM        │
 └──────────────────────┘          │ → grounded answer         │
                                   └──────────────────────────┘
```

| Piece | Choice | Notes |
|---|---|---|
| Vector DB | **sqlite-vec** `vec0` table in `rag.db` | the thing under test |
| Embeddings | `BAAI/bge-small-en-v1.5` via [fastembed](https://github.com/qdrant/fastembed) | runs locally, 384-dim, no API key, downloads ~90 MB once |
| LLM | Groq `openai/gpt-oss-120b` | any Groq chat model works — edit `MODEL` in `app.py` |
| Web framework | FastAPI + uvicorn | 3 routes: `/`, `/upload`, `/chat` |
| UI | one static [index.html](index.html) | file picker + chat, no build step |

Chunking is deliberately simple: fixed ~1000-character windows with 200-character overlap. The
point was to test the vector store, not to tune a retrieval pipeline.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows  (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
echo GROQ_API_KEY=gsk_your_key_here > .env
```

Get a free Groq key at <https://console.groq.com>. The app reads `.env` on startup (via
`python-dotenv`); `.env` is git-ignored.

## Run

```bash
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>, upload a `.txt`, `.md`, `.pdf`, `.csv`, `.json`, or `.log` file,
then ask questions about it.

`rag.db` persists between runs — delete it to start with an empty knowledge base.

---

## Did `sqlite-vec` hold up? — honest notes

**Yes, comfortably, for this scale.**

What went well:

- **Install** — `pip install sqlite-vec`, one wheel, no native build, worked first try on Windows
  with the stock python.org interpreter.
- **API** — it's just SQL. `CREATE VIRTUAL TABLE ... USING vec0`, `INSERT`, `SELECT ... WHERE
  embedding MATCH ? AND k = ?`. Nothing to learn beyond that.
- **Auxiliary (`+`) columns** — being able to keep the chunk text and its source filename *in the
  same row as the vector* and get them straight back from the KNN query removed an entire layer of
  bookkeeping.
- **Retrieval quality** — every test question pulled back the relevant chunks. Grounded answers were
  correct; questions with no supporting text correctly produced "I don't know."
- **Operationally trivial** — the entire vector store is one `rag.db` file. Copy it, commit it,
  delete it. No server, no container, no connection string.

Things to be aware of (limitations of the tool, not bugs):

- The default `vec0` index is a **brute-force / linear scan**. It's genuinely fast for thousands to
  low millions of vectors; if you're at tens of millions with strict latency targets, benchmark
  before committing (ANN indexing is on the project's roadmap).
- `enable_load_extension` has to be available in your Python build. Fine on standard Windows and
  Linux CPython; some locked-down or system Python builds (notably older macOS system Python)
  disable it.
- It's a younger project (pre-1.0) than pgvector or FAISS. It moved fast and did everything this
  project asked of it, but that's the maturity trade-off to weigh for a large production system.

**Bottom line:** for embedded, single-file, up-to-millions-of-vectors retrieval, `sqlite-vec` is an
easy yes. This chatbot never had to work around it once.

---

## Project structure

```
app.py             backend — extension loading, chunking, embedding, the 3 routes
index.html         the entire UI
requirements.txt   dependencies
.env               GROQ_API_KEY (git-ignored, create this yourself)
rag.db             the sqlite-vec database (created on first upload, git-ignored)
```
