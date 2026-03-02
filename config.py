# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load env from this project folder regardless of current working directory.
_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_BASE_DIR / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# ── API Keys ──────────────────────────────
# ❌ Removed: OPENAI_API_KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip() or None  # ✅ Gemini only
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "").strip() or None     # ✅ Groq for open-source

# ── Supported Domains ─────────────────────
DOMAINS = ["medical", "legal", "finance", "technology", "general"]

# ── Model Routing ─────────────────────────
# ✅ All API-based models now use Gemini
# ✅ Open-source models use Groq (free)
DOMAIN_MODEL_MAP = {
    "medical":    {"provider": "gemini", "model": "gemini-2.5-pro"},
    "legal":      {"provider": "gemini", "model": "gemini-2.5-pro"},
    "finance":    {"provider": "gemini", "model": "gemini-2.5-flash"},
    "technology": {"provider": "groq",   "model": "mistral"},
    "general":    {"provider": "groq",   "model": "llama3"},
}

# ── Embedding ─────────────────────────────
# ✅ Switched from OpenAI → Gemini embeddings
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_PROVIDER = "gemini"

# ── Chroma ─────────────────────────────────
CHROMA_ANONYMIZED_TELEMETRY = _env_bool("CHROMA_ANONYMIZED_TELEMETRY", False)

# ── Chunking ──────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# ── Retrieval ─────────────────────────────
TOP_K = 5

# ── VectorStore ───────────────────────────
CHROMA_BASE_DIR = "./chroma_stores"

# ── LLM Settings ──────────────────────────
TEMPERATURE = 0.2
MAX_TOKENS  = 1024
