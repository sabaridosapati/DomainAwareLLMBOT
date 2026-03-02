# rag/vectorstore.py
# ✅ Replaced OpenAIEmbeddings → GoogleGenerativeAIEmbeddings

import os
import logging
from typing import List

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # ✅ Changed
from langchain.schema import Document

from config import (
    GOOGLE_API_KEY,       # ✅ Changed from OPENAI_API_KEY
    EMBEDDING_MODEL,
    CHROMA_ANONYMIZED_TELEMETRY,
    CHROMA_BASE_DIR,
    TOP_K
)

logger = logging.getLogger(__name__)


def _resolve_embedding_model(model_name: str) -> str:
    deprecated = {"models/embedding-001", "models/text-embedding-004"}
    if model_name in deprecated:
        fallback = "models/gemini-embedding-001"
        logger.warning(
            "Embedding model '%s' is incompatible with this SDK/API path. "
            "Falling back to '%s'.",
            model_name,
            fallback,
        )
        return fallback
    return model_name


class DomainVectorStore:
    """Separate ChromaDB collection per knowledge domain."""

    def __init__(self, domain: str):
        self.domain      = domain
        self.persist_dir = os.path.join(CHROMA_BASE_DIR, domain)
        telemetry_impl = (
            "chromadb.telemetry.product.posthog.Posthog"
            if CHROMA_ANONYMIZED_TELEMETRY
            else "rag.noop_telemetry.NoOpTelemetry"
        )
        self.client_settings = Settings(
            is_persistent=True,
            persist_directory=self.persist_dir,
            anonymized_telemetry=CHROMA_ANONYMIZED_TELEMETRY,
            chroma_product_telemetry_impl=telemetry_impl,
            chroma_telemetry_impl=telemetry_impl,
        )

        # ✅ Replaced OpenAIEmbeddings → GoogleGenerativeAIEmbeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=_resolve_embedding_model(EMBEDDING_MODEL),
            google_api_key=GOOGLE_API_KEY,
            task_type="retrieval_document"
        )
        self.vectorstore = None

    def create(self, chunks: List[Document]) -> Chroma:
        """Create a new vectorstore from document chunks."""
        logger.info(
            f"🆕 Creating vectorstore for '{self.domain}' "
            f"with {len(chunks)} chunks..."
        )
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client_settings=self.client_settings,
            persist_directory=self.persist_dir,
            collection_name=f"{self.domain}_knowledge"
        )
        logger.info(f"✅ Vectorstore created for domain: '{self.domain}'")
        return self.vectorstore

    def load(self) -> Chroma:
        """Load existing vectorstore from disk."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            client_settings=self.client_settings,
            collection_name=f"{self.domain}_knowledge"
        )
        logger.info(f"✅ Vectorstore loaded for domain: '{self.domain}'")
        return self.vectorstore

    def add_documents(self, chunks: List[Document]):
        """Add new documents to existing vectorstore."""
        if not self.vectorstore:
            raise RuntimeError("Vectorstore not initialized. Call load() first.")
        self.vectorstore.add_documents(chunks)
        logger.info(f"➕ Added {len(chunks)} chunks to '{self.domain}'")

    def retrieve(self, query: str, k: int = TOP_K) -> List[Document]:
        """Retrieve top-k similar chunks for a query."""
        if not self.vectorstore:
            self.load()
        return self.vectorstore.similarity_search(query, k=k)
