# model_router.py
# ✅ Removed OpenAI, now routes between Gemini and Groq only

import logging
from config import DOMAIN_MODEL_MAP
from models.gemini_model import GeminiModel
from models.groq_model import GroqModel     
# ❌ Removed: from models.openai_model import OpenAIModel

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Routes queries between:
    - Gemini 2.5 Pro/Flash  → Medical, Legal, Finance
    - Groq (Mistral/LLaMA)  → Technology, General
    """

    def __init__(self):
        self._model_cache = {}

    def get_model(self, domain: str):
        """Return the appropriate model for a given domain."""
        config     = DOMAIN_MODEL_MAP.get(domain, DOMAIN_MODEL_MAP["general"])
        provider   = config["provider"]
        model_name = config["model"]
        cache_key  = f"{provider}:{model_name}"

        if cache_key not in self._model_cache:
            if provider == "gemini":
                self._model_cache[cache_key] = GeminiModel(model_name)
            elif provider == "groq":
                try:
                    self._model_cache[cache_key] = GroqModel(model_name)
                except Exception as e:
                    fallback_provider = "gemini"
                    fallback_model = "gemini-2.5-flash"
                    fallback_key = f"{fallback_provider}:{fallback_model}"
                    logger.warning(
                        "Failed to initialize Groq model '%s' (%s). "
                        "Falling back to %s (%s).",
                        model_name,
                        e,
                        fallback_provider,
                        fallback_model,
                    )
                    if fallback_key not in self._model_cache:
                        self._model_cache[fallback_key] = GeminiModel(fallback_model)
                    return (
                        self._model_cache[fallback_key],
                        fallback_provider,
                        fallback_model,
                    )
            else:
                raise ValueError(f"Unknown provider: {provider}")

        model = self._model_cache[cache_key]
        logger.info(f"Domain '{domain}' → {provider} ({model_name})")
        return model, provider, model_name
