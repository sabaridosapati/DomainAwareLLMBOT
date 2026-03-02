# domain_classifier.py
# ✅ Replaced GPT-4o-mini → Gemini Flash for classification

import logging
from langchain_google_genai import ChatGoogleGenerativeAI   # ✅ Changed
from langchain.prompts import PromptTemplate
from config import GOOGLE_API_KEY, DOMAINS                  # ✅ Removed OPENAI

logger = logging.getLogger(__name__)

DOMAIN_KEYWORDS = {
    "medical":    ["symptom", "disease", "treatment", "doctor", "medicine",
                   "diagnosis", "patient", "hospital", "drug", "health"],
    "legal":      ["law", "contract", "rights", "court", "attorney", "clause",
                   "liability", "sue", "legal", "regulation", "compliance"],
    "finance":    ["investment", "stock", "market", "budget", "tax", "loan",
                   "revenue", "profit", "bank", "portfolio", "crypto"],
    "technology": ["code", "software", "api", "python", "bug", "database",
                   "cloud", "algorithm", "machine learning", "ai", "server"],
}

CLASSIFY_PROMPT = PromptTemplate(
    input_variables=["query", "domains"],
    template="""
You are a domain classifier. Given a user query, classify it into ONE of these domains:
{domains}

Rules:
- Return ONLY the domain name, nothing else
- If unsure, return "general"

Query: {query}
Domain:"""
)


class DomainClassifier:

    def __init__(self):
        # ✅ Replaced ChatOpenAI → ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",       # Fast + cheap for classification
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True
        )

    def classify_keyword(self, query: str) -> str:
        """Fast keyword-based classification — no API call needed."""
        query_lower = query.lower()
        scores = {domain: 0 for domain in DOMAIN_KEYWORDS}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[domain] += 1

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def classify_llm(self, query: str) -> str:
        """Gemini-based classification — more accurate."""
        prompt = CLASSIFY_PROMPT.format(
            query=query,
            domains=", ".join(DOMAINS)
        )
        response = self.llm.invoke(prompt)
        domain = response.content.strip().lower()
        return domain if domain in DOMAINS else "general"

    def classify(self, query: str, use_llm: bool = False) -> str:
        """
        Classify query domain.
        - Keyword-based by default (fast, no API cost)
        - LLM-based for higher accuracy
        """
        if use_llm:
            domain = self.classify_llm(query)
        else:
            domain = self.classify_keyword(query)

        logger.info(f"Query classified as domain: '{domain}'")
        return domain