# rag/document_loader.py

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Handles loading documents from multiple sources:
    - PDF files
    - TXT / Markdown files
    - CSV files
    - Web URLs
    - Entire directories
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    # ── Loaders ──────────────────────────────────────────────────────

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF file page by page."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info(f"📄 Loaded PDF: {file_path} ({len(docs)} pages)")
        return docs

    def load_text(self, file_path: str) -> List[Document]:
        """Load a plain text or markdown file."""
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        logger.info(f"📄 Loaded Text: {file_path} ({len(docs)} docs)")
        return docs

    def load_csv(self, file_path: str) -> List[Document]:
        """Load a CSV file — each row becomes a document."""
        loader = CSVLoader(file_path)
        docs = loader.load()
        logger.info(f"📄 Loaded CSV: {file_path} ({len(docs)} rows)")
        return docs

    def load_web(self, urls: List[str]) -> List[Document]:
        """Load content from web URLs."""
        loader = WebBaseLoader(urls)
        docs = loader.load()
        logger.info(f"🌐 Loaded {len(docs)} web pages")
        return docs

    def load_directory(self, dir_path: str,
                        glob_pattern: str = "**/*.pdf") -> List[Document]:
        """Load all matching files from a directory."""
        loader = DirectoryLoader(
            dir_path,
            glob=glob_pattern,
            loader_cls=PyPDFLoader
        )
        docs = loader.load()
        logger.info(f"📁 Loaded {len(docs)} docs from directory: {dir_path}")
        return docs

    def load_file(self, file_path: str) -> List[Document]:
        """
        Auto-detect file type and load accordingly.
        Supports: .pdf, .txt, .md, .csv
        """
        path = Path(file_path)
        ext  = path.suffix.lower()

        loaders = {
            ".pdf": self.load_pdf,
            ".txt": self.load_text,
            ".md":  self.load_text,
            ".csv": self.load_csv,
        }

        loader_fn = loaders.get(ext)
        if not loader_fn:
            raise ValueError(
                f"Unsupported file type: '{ext}'. "
                f"Supported: {list(loaders.keys())}"
            )

        return loader_fn(file_path)

    # ── Chunking ─────────────────────────────────────────────────────

    def chunk(self, docs: List[Document]) -> List[Document]:
        """Split documents into overlapping chunks with enriched metadata."""
        chunks = self.splitter.split_documents(docs)

        # Enrich each chunk with extra metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"]  = len(chunk.page_content)
            chunk.metadata["word_count"]  = len(chunk.page_content.split())

        logger.info(f"✂️  Created {len(chunks)} chunks")
        return chunks

    # ── Full Pipeline ─────────────────────────────────────────────────

    def process_file(self, file_path: str) -> List[Document]:
        """
        Full pipeline for a single file:
        Load → Chunk → Return
        """
        docs   = self.load_file(file_path)
        chunks = self.chunk(docs)
        logger.info(
            f"✅ Processed '{Path(file_path).name}': "
            f"{len(docs)} pages → {len(chunks)} chunks"
        )
        return chunks

    def process_directory(self, dir_path: str,
                           domain: str = "general") -> List[Document]:
        """
        Full pipeline for an entire directory.
        Adds domain tag to each chunk's metadata.
        """
        docs   = self.load_directory(dir_path)
        chunks = self.chunk(docs)

        # Tag each chunk with domain
        for chunk in chunks:
            chunk.metadata["domain"] = domain

        logger.info(
            f"✅ Processed directory '{dir_path}': "
            f"{len(docs)} docs → {len(chunks)} chunks"
        )
        return chunks

    def process_urls(self, urls: List[str]) -> List[Document]:
        """Full pipeline for web URLs: Load → Chunk → Return."""
        docs   = self.load_web(urls)
        chunks = self.chunk(docs)
        logger.info(f"✅ Processed {len(urls)} URLs → {len(chunks)} chunks")
        return chunks


# ── Quick Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DocumentLoader()

    # Test with a sample file
    import sys
    if len(sys.argv) > 1:
        chunks = loader.process_file(sys.argv[1])
        print(f"\nSample chunk:\n{chunks[0].page_content[:300]}")
        print(f"Metadata: {chunks[0].metadata}")