# rag/retriever.py

import logging
from typing import List, Optional
from langchain.schema import Document
from rag.vectorstore import DomainVectorStore
from config import TOP_K

logger = logging.getLogger(__name__)


class DomainRetriever:
    """
    Handles retrieval strategies for domain-specific knowledge:
    - Similarity search
    - MMR (Maximal Marginal Relevance) for diverse results
    - Score threshold filtering
    - Hybrid: keyword + semantic
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.vs = DomainVectorStore(domain)
        try:
            self.vs.load()
            logger.info(f"✅ Retriever ready for domain: '{domain}'")
        except Exception as e:
            logger.warning(
                f"⚠️  No vectorstore found for domain '{domain}'. "
                f"Please ingest documents first. Error: {e}"
            )

    # ── Retrieval Strategies ─────────────────────────────────────────

    def similarity_search(self, query: str,
                           k: int = TOP_K) -> List[Document]:
        """
        Basic cosine similarity search.
        Fast, good for most use cases.
        """
        try:
            docs = self.vs.vectorstore.similarity_search(query, k=k)
            logger.info(
                f"🔍 Similarity search: '{query[:50]}...' "
                f"→ {len(docs)} results"
            )
            return docs
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def mmr_search(self, query: str, k: int = TOP_K,
                   fetch_k: int = None,
                   lambda_mult: float = 0.7) -> List[Document]:
        """
        Maximal Marginal Relevance search.
        Balances relevance + diversity — avoids redundant chunks.

        lambda_mult: 0 = max diversity, 1 = max relevance
        """
        fetch_k = fetch_k or k * 3
        try:
            docs = self.vs.vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            logger.info(
                f"🔍 MMR search: '{query[:50]}...' "
                f"→ {len(docs)} diverse results"
            )
            return docs
        except Exception as e:
            logger.error(f"MMR search failed: {e}")
            return []

    def threshold_search(self, query: str,
                          score_threshold: float = 0.75,
                          k: int = TOP_K) -> List[Document]:
        """
        Only return results above a confidence threshold.
        Useful for avoiding low-quality retrievals.
        """
        try:
            retriever = self.vs.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": score_threshold,
                    "k": k
                }
            )
            docs = retriever.invoke(query)
            logger.info(
                f"🔍 Threshold search (>{score_threshold}): "
                f"'{query[:50]}...' → {len(docs)} results"
            )
            return docs
        except Exception as e:
            logger.error(f"Threshold search failed: {e}")
            return []

    def hybrid_search(self, query: str,
                       k: int = TOP_K) -> List[Document]:
        """
        Hybrid search: combines semantic + keyword matching.
        Deduplicates and re-ranks results.
        Best for precise domain queries.
        """
        # Semantic results
        semantic_docs = self.similarity_search(query, k=k)

        # MMR results for diversity
        mmr_docs = self.mmr_search(query, k=k)

        # Merge + deduplicate
        seen_contents = set()
        combined = []
        for doc in semantic_docs + mmr_docs:
            content_key = doc.page_content[:100]  # First 100 chars as key
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                combined.append(doc)

        # Return top-k after dedup
        result = combined[:k]
        logger.info(
            f"🔍 Hybrid search: '{query[:50]}...' "
            f"→ {len(result)} unique results"
        )
        return result

    # ── Main Retrieve Method ─────────────────────────────────────────

    def retrieve(self, query: str,
                 strategy: str = "mmr",
                 k: int = TOP_K) -> List[Document]:
        """
        Main retrieval entry point.

        Strategies:
        - 'similarity'  → Fast cosine similarity
        - 'mmr'         → Diverse results (recommended)
        - 'threshold'   → High confidence only
        - 'hybrid'      → Best quality, slightly slower
        """
        strategies = {
            "similarity": self.similarity_search,
            "mmr":        self.mmr_search,
            "threshold":  self.threshold_search,
            "hybrid":     self.hybrid_search
        }

        search_fn = strategies.get(strategy)
        if not search_fn:
            logger.warning(
                f"Unknown strategy '{strategy}', "
                f"falling back to 'similarity'"
            )
            search_fn = self.similarity_search

        return search_fn(query, k=k)

    # ── Context Builder ──────────────────────────────────────────────

    def get_context(self, query: str,
                    strategy: str = "mmr",
                    k: int = TOP_K) -> dict:
        """
        Retrieve docs and format them as context for the LLM prompt.
        Returns both raw docs and formatted context string.
        """
        docs = self.retrieve(query, strategy=strategy, k=k)

        if not docs:
            return {
                "context_str": "No relevant context found.",
                "sources":     [],
                "docs":        []
            }

        # Format context string for prompt injection
        context_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            context_parts.append(
                f"[{i}] Source: {meta.get('source', 'Unknown')} "
                f"| Page: {meta.get('page', 'N/A')} "
                f"| Domain: {meta.get('domain', self.domain)}\n"
                f"{doc.page_content}"
            )

        context_str = "\n\n---\n\n".join(context_parts)

        # Extract unique sources
        sources = []
        seen = set()
        for doc in docs:
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", "")
            )
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page":   doc.metadata.get("page", "N/A"),
                    "domain": doc.metadata.get("domain", self.domain)
                })

        return {
            "context_str": context_str,
            "sources":     sources,
            "docs":        docs
        }


# ── Multi-Domain Retriever ───────────────────────────────────────────

class MultiDomainRetriever:
    """
    Retrieves across ALL domains simultaneously.
    Useful when domain is uncertain or for cross-domain queries.
    """

    def __init__(self, domains: List[str]):
        self.retrievers = {}
        for domain in domains:
            try:
                self.retrievers[domain] = DomainRetriever(domain)
            except Exception as e:
                logger.warning(f"Could not load retriever for '{domain}': {e}")

    def retrieve_all(self, query: str,
                     k_per_domain: int = 2) -> List[Document]:
        """Search across all domains and merge results."""
        all_docs = []
        for domain, retriever in self.retrievers.items():
            docs = retriever.retrieve(query, k=k_per_domain)
            # Tag domain in metadata
            for doc in docs:
                doc.metadata["domain"] = domain
            all_docs.extend(docs)

        logger.info(
            f"🌐 Multi-domain search: "
            f"{len(all_docs)} total results across "
            f"{len(self.retrievers)} domains"
        )
        return all_docs


# ── Quick Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    retriever = DomainRetriever("technology")
    result = retriever.get_context(
        "What is machine learning?",
        strategy="mmr"
    )

    print(f"\nContext:\n{result['context_str'][:500]}")
    print(f"\nSources: {result['sources']}")
