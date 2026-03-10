
# Domain-Aware LLM Chatbot

A Streamlit chatbot that:
- Classifies each query into a domain (`medical`, `legal`, `finance`, `technology`, `general`)
- Routes the query to the best LLM per domain (Gemini or Groq)
- Retrieves supporting context from domain-specific Chroma vector stores (RAG)
- Shows model/domain metadata and document sources in the UI

Project code lives in [`llmchatbot/`](./llmchatbot).

## Features

- Domain-aware routing:
  - `medical` -> `gemini-2.5-pro`
  - `legal` -> `gemini-2.5-pro`
  - `finance` -> `gemini-2.5-flash`
  - `technology` -> Groq-hosted OSS model
  - `general` -> Groq-hosted OSS model
- RAG ingestion for PDF, TXT, MD, CSV, and URLs
- Domain-specific Chroma collections under `chroma_stores/<domain>/`
- Session/domain conversation memory (windowed)
- Streamlit UI with source citations

## Project Structure

```text
llmchatbot/
  ui.py                     # Streamlit app entrypoint
  chatbot.py                # Main orchestration logic
  domain_classifier.py      # Keyword or LLM-based domain classifier
  model_router.py           # Domain -> provider/model routing
  ingest.py                 # CLI ingestion tool
  config.py                 # Runtime config + env loading
  rag/
    document_loader.py      # Load/chunk docs and URLs
    vectorstore.py          # Chroma + Gemini embeddings
  memory/
    conversation_memory.py  # Per-session/domain memory
  docs/                     # Domain folders for source docs
  chroma_stores/            # Persisted vector stores
```

## Requirements

- Python 3.11+ recommended
- API keys:
  - `GOOGLE_API_KEY` (required)
  - `GROQ_API_KEY` (recommended for technology/general routes)

## Setup

From the repository root:

```powershell
cd llmchatbot
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `llmchatbot/.env`:

```env
GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key

# Optional
EMBEDDING_MODEL=models/gemini-embedding-001
CHROMA_ANONYMIZED_TELEMETRY=false
```

## Ingest Documents (RAG)

Run from `llmchatbot/`.

1. Ingest all default domain folders (`docs/<domain>`):

```powershell
python ingest.py all
```

2. Or ingest one file:

```powershell
python ingest.py file --path docs\medical\your_file.pdf --domain medical
```

3. Or ingest a directory:

```powershell
python ingest.py directory --path docs\finance --domain finance
```

4. Check index status:

```powershell
python ingest.py status
```

5. Clear one domain index:

```powershell
python ingest.py clear --domain legal
```

## Run the App

From `llmchatbot/`:

```powershell
streamlit run ui.py
```

Then open the local Streamlit URL shown in the terminal (usually `http://localhost:8501`).

## How It Works

1. Query is classified by domain (`keyword` by default, optional Gemini classifier in UI toggle).
2. Domain vector store retrieves top matching chunks (`TOP_K` in `config.py`).
3. Domain-specific system prompt is built with retrieved context.
4. Router picks LLM provider/model for that domain.
5. Response and source metadata are returned to UI and saved in session memory.

## Configuration

Key settings are in `llmchatbot/config.py`:

- `DOMAIN_MODEL_MAP` for routing
- `CHUNK_SIZE`, `CHUNK_OVERLAP` for splitting
- `TOP_K` for retrieval depth
- `TEMPERATURE`, `MAX_TOKENS` for generation
- `CHROMA_BASE_DIR` for vector store location

## Troubleshooting

- `Missing GROQ_API_KEY`:
  - Add `GROQ_API_KEY` to `llmchatbot/.env`, or adjust routing in `DOMAIN_MODEL_MAP`.
- No sources returned:
  - Run ingestion first (`python ingest.py all`) and verify with `python ingest.py status`.
- Chroma telemetry warnings:
  - Set `CHROMA_ANONYMIZED_TELEMETRY=false` in `.env`.

## Notes

- This repo currently includes a local virtual environment and generated artifacts inside `llmchatbot/` (`Lib/`, `Scripts/`, `__pycache__/`, `chroma_stores/`). For cleaner version control, consider adding/expanding `.gitignore`.

## Next Improvements

- Add automated tests for routing, classification, and ingestion CLI to prevent regressions.
- Use a hybrid retriever (vector + keyword/BM25) and reranking for better context quality.
- Add confidence scoring for domain classification, with fallback to `general` on low confidence.
- Introduce response streaming in the Streamlit UI for faster perceived latency.
- Add document management in UI (upload, re-index, delete) so ingestion is not CLI-only.
- Add observability (request logs, latency, token usage, failure rates) for production monitoring.
