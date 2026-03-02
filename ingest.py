# ingest.py
# Run this script to load documents into the RAG system domain by domain
# Usage: python ingest.py

import os
import logging
import argparse
from pathlib import Path
from typing import List

from rag.document_loader import DocumentLoader
from rag.vectorstore import DomainVectorStore
from config import DOMAINS, CHROMA_BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Domain → Default Folder Mapping ─────────────────────────────────
DOMAIN_FOLDERS = {
    "medical":    "docs/medical",
    "legal":      "docs/legal",
    "finance":    "docs/finance",
    "technology": "docs/technology",
    "general":    "docs/general"
}


# ── Single File Ingestion ────────────────────────────────────────────

def ingest_file(file_path: str, domain: str) -> bool:
    """
    Ingest a single file into a specific domain vectorstore.

    Args:
        file_path : Path to the file (PDF, TXT, CSV, MD)
        domain    : Domain to store it under

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"📄 Ingesting file: '{file_path}' → domain: '{domain}'")

    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        return False

    if domain not in DOMAINS:
        logger.error(
            f"❌ Invalid domain: '{domain}'. "
            f"Valid domains: {DOMAINS}"
        )
        return False

    try:
        # Step 1 - Load & Chunk
        loader = DocumentLoader()
        chunks = loader.process_file(file_path)

        if not chunks:
            logger.warning(f"⚠️  No chunks created from: {file_path}")
            return False

        # Tag each chunk with domain metadata
        for chunk in chunks:
            chunk.metadata["domain"] = domain

        # Step 2 - Store in VectorDB
        vs = DomainVectorStore(domain)
        persist_dir = os.path.join(CHROMA_BASE_DIR, domain)

        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            # Vectorstore exists → add to it
            logger.info(f"📦 Adding to existing '{domain}' vectorstore...")
            vs.load()
            vs.add_documents(chunks)
        else:
            # Fresh vectorstore → create new
            logger.info(f"🆕 Creating new '{domain}' vectorstore...")
            vs.create(chunks)

        logger.info(
            f"✅ Successfully ingested '{Path(file_path).name}': "
            f"{len(chunks)} chunks → '{domain}' domain"
        )
        return True

    except Exception as e:
        logger.error(f"❌ Ingestion failed for '{file_path}': {e}")
        return False


# ── Directory Ingestion ──────────────────────────────────────────────

def ingest_directory(dir_path: str, domain: str) -> dict:
    """
    Ingest ALL supported files from a directory into a domain.

    Args:
        dir_path : Path to the directory
        domain   : Domain to store under

    Returns:
        Summary dict with success/failure counts
    """
    logger.info(f"📁 Ingesting directory: '{dir_path}' → domain: '{domain}'")

    if not os.path.exists(dir_path):
        logger.error(f"❌ Directory not found: {dir_path}")
        return {"success": 0, "failed": 0, "skipped": 0}

    # Find all supported files
    supported_extensions = [".pdf", ".txt", ".md", ".csv"]
    files = []
    for ext in supported_extensions:
        files.extend(Path(dir_path).rglob(f"*{ext}"))

    if not files:
        logger.warning(f"⚠️  No supported files found in: {dir_path}")
        return {"success": 0, "failed": 0, "skipped": 0}

    logger.info(f"🔎 Found {len(files)} files to process...")

    results = {"success": 0, "failed": 0, "skipped": 0}

    for file_path in files:
        success = ingest_file(str(file_path), domain)
        if success:
            results["success"] += 1
        else:
            results["failed"] += 1

    logger.info(
        f"\n📊 Directory Ingestion Summary for '{domain}':\n"
        f"   ✅ Success : {results['success']}\n"
        f"   ❌ Failed  : {results['failed']}\n"
        f"   ⏭️  Skipped : {results['skipped']}"
    )
    return results


# ── URL Ingestion ────────────────────────────────────────────────────
#urls: List[www.google.com], domain: str
#urls= {'www.google.com', 'www.groq.com'}
#domain= 'Technology'

def ingest_urls(urls: List[str], domain: str) -> bool:
    """
    Ingest content from web URLs into a domain.

    Args:
        urls   : List of URLs to scrape
        domain : Domain to store under
    """
    logger.info(f"🌐 Ingesting {len(urls)} URLs → domain: '{domain}'")

    try:
        loader = DocumentLoader()
        chunks = loader.process_urls(urls)

        if not chunks:
            logger.warning("⚠️  No content retrieved from URLs")
            return False

        for chunk in chunks:
            chunk.metadata["domain"] = domain

        vs = DomainVectorStore(domain)
        persist_dir = os.path.join(CHROMA_BASE_DIR, domain)

        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            vs.load()
            vs.add_documents(chunks)
        else:
            vs.create(chunks)

        logger.info(
            f"✅ Ingested {len(urls)} URLs: "
            f"{len(chunks)} chunks → '{domain}' domain"
        )
        return True

    except Exception as e:
        logger.error(f"❌ URL ingestion failed: {e}")
        return False


# ── Bulk Ingest All Domains ──────────────────────────────────────────

def ingest_all_domains():
    """
    Ingest ALL domain folders in one go.
    Reads from the DOMAIN_FOLDERS mapping in config.
    """
    logger.info("🚀 Starting bulk ingestion for ALL domains...")
    overall = {"success": 0, "failed": 0}

    for domain, folder in DOMAIN_FOLDERS.items():
        if os.path.exists(folder):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing domain: {domain.upper()}")
            logger.info(f"{'='*50}")
            results = ingest_directory(folder, domain)
            overall["success"] += results["success"]
            overall["failed"]  += results["failed"]
        else:
            logger.warning(
                f"⚠️  Folder not found for '{domain}': '{folder}' — skipping"
            )

    logger.info(
        f"\n{'='*50}\n"
        f"🏁 BULK INGESTION COMPLETE\n"
        f"   ✅ Total Success : {overall['success']}\n"
        f"   ❌ Total Failed  : {overall['failed']}\n"
        f"{'='*50}"
    )


# ── Vectorstore Status Check ─────────────────────────────────────────

def check_status():
    """Check ingestion status for all domains."""
    logger.info("\n📊 Vectorstore Status Check:")
    logger.info(f"{'='*50}")

    for domain in DOMAINS:
        persist_dir = os.path.join(CHROMA_BASE_DIR, domain)
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            try:
                vs = DomainVectorStore(domain)
                vs.load()
                count = vs.vectorstore._collection.count()
                logger.info(f"  ✅ {domain:15} → {count} chunks indexed")
            except Exception as e:
                logger.info(f"  ⚠️  {domain:15} → Error reading: {e}")
        else:
            logger.info(f"  ❌ {domain:15} → Not ingested yet")

    logger.info(f"{'='*50}\n")


# ── Clear a Domain ───────────────────────────────────────────────────

def clear_domain(domain: str):
    """Delete and reset a domain's vectorstore."""
    import shutil
    persist_dir = os.path.join(CHROMA_BASE_DIR, domain)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        logger.info(f"🗑️  Cleared vectorstore for domain: '{domain}'")
    else:
        logger.warning(f"⚠️  No vectorstore found for domain: '{domain}'")


# ── CLI Interface ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="📚 Domain-Aware Chatbot — Document Ingestion Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── ingest-file command
    file_parser = subparsers.add_parser(
        "file", help="Ingest a single file"
    )
    file_parser.add_argument("--path",   required=True, help="Path to file")
    file_parser.add_argument("--domain", required=True,
                              choices=DOMAINS, help="Target domain")

    # ── ingest-dir command
    dir_parser = subparsers.add_parser(
        "directory", help="Ingest all files in a directory"
    )
    dir_parser.add_argument("--path",   required=True, help="Directory path")
    dir_parser.add_argument("--domain", required=True,
                             choices=DOMAINS, help="Target domain")

    # ── ingest-urls command
    url_parser = subparsers.add_parser(
        "urls", help="Ingest content from URLs"
    )
    url_parser.add_argument("--urls",   nargs="+", required=True,
                             help="Space-separated list of URLs")
    url_parser.add_argument("--domain", required=True,
                             choices=DOMAINS, help="Target domain")

    # ── ingest-all command
    subparsers.add_parser(
        "all", help="Ingest all domain folders at once"
    )

    # ── status command
    subparsers.add_parser(
        "status", help="Check ingestion status for all domains"
    )

    # ── clear command
    clear_parser = subparsers.add_parser(
        "clear", help="Clear a domain's vectorstore"
    )
    clear_parser.add_argument("--domain", required=True,
                               choices=DOMAINS, help="Domain to clear")

    # ── Parse & Execute
    args = parser.parse_args()

    if args.command == "file":
        ingest_file(args.path, args.domain)

    elif args.command == "directory":
        ingest_directory(args.path, args.domain)

    elif args.command == "urls":
        ingest_urls(args.urls, args.domain)

    elif args.command == "all":
        ingest_all_domains()

    elif args.command == "status":
        check_status()

    elif args.command == "clear":
        clear_domain(args.domain)


if __name__ == "__main__":
    main()