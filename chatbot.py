# chatbot.py
import logging
from domain_classifier import DomainClassifier
from model_router import ModelRouter
from rag.vectorstore import DomainVectorStore
from memory.conversation_memory import ConversationMemoryManager

logger = logging.getLogger(__name__)

# Domain-specific system prompts
DOMAIN_PROMPTS = {
    "medical": """You are a medical information assistant. Provide accurate, 
evidence-based information. Always recommend consulting a licensed physician 
for personal medical decisions. Use the context below to answer.""",

    "legal": """You are a legal information assistant. Provide clear explanations 
of legal concepts. Always recommend consulting a licensed attorney for specific 
legal advice. Use the context below to answer.""",

    "finance": """You are a financial information assistant. Provide data-driven 
financial insights. Always recommend consulting a financial advisor for 
investment decisions. Use the context below to answer.""",

    "technology": """You are an expert software engineer and technical assistant. 
Provide clear, accurate technical explanations with code examples when helpful. 
Use the context below to answer.""",

    "general": """You are a helpful, knowledgeable assistant. 
Answer accurately and concisely using the context below."""
}


class DomainAwareChatbot:

    def __init__(self):
        self.classifier  = DomainClassifier()
        self.router      = ModelRouter()
        self.memory_mgr  = ConversationMemoryManager(window_size=5)
        self.vs_cache    = {}   # domain → DomainVectorStore

    def _get_vectorstore(self, domain: str) -> DomainVectorStore:
        if domain not in self.vs_cache:
            vs = DomainVectorStore(domain)
            try:
                vs.load()
                self.vs_cache[domain] = vs
            except Exception:
                logger.warning(f"No vectorstore found for domain: {domain}")
                return None
        return self.vs_cache.get(domain)

    def _build_augmented_prompt(self, domain: str,
                                 query: str, context_docs: list) -> str:
        system_prompt = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])

        if context_docs:
            context = "\n\n---\n".join([
                f"[Source: {d.metadata.get('source','N/A')}, "
                f"Page: {d.metadata.get('page','N/A')}]\n{d.page_content}"
                for d in context_docs
            ])
            system_prompt += f"\n\nCONTEXT:\n{context}"

        return system_prompt

    def chat(self, query: str, session_id: str = "default",
             use_llm_classifier: bool = False) -> dict:
        # 1. Classify domain
        domain = self.classifier.classify(query, use_llm=use_llm_classifier)

        # 2. Retrieve relevant context
        vs = self._get_vectorstore(domain)
        context_docs = vs.retrieve(query) if vs else []

        # 3. Get chat history
        history = self.memory_mgr.get_history(session_id, domain)

        # 4. Build augmented prompt
        system_prompt = self._build_augmented_prompt(domain, query, context_docs)

        # 5. Route to correct model and generate
        model, provider, model_name = self.router.get_model(domain)
        answer = model.generate(system_prompt, query)

        # 6. Save to memory
        self.memory_mgr.save_turn(session_id, domain, query, answer)

        # 7. Return full response
        return {
            "answer": answer,
            "domain": domain,
            "model_used": f"{provider} / {model_name}",
            "sources": [
                {"source": d.metadata.get("source", "N/A"),
                 "page":   d.metadata.get("page", "N/A")}
                for d in context_docs
            ],
            "context_chunks": len(context_docs)
        }