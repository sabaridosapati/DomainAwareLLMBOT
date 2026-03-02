# models/groq_model.py
# pip install groq langchain-groq

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from groq import BadRequestError
from config import GROQ_API_KEY, TEMPERATURE, MAX_TOKENS

class GroqModel:
    """
    Groq-hosted open-source models.
    Free tier: 14,400 req/day
    Models: Mistral, LLaMA3, Gemma
    """

    AVAILABLE_MODELS = {
        # Keep old aliases for backwards compatibility in routing config.
        "mistral": "llama-3.1-8b-instant",
        "llama3": "llama-3.1-8b-instant",
        "llama3-70b": "llama-3.3-70b-versatile",
        "gemma": "llama-3.1-8b-instant",
        # Explicit modern Groq model ids.
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
        "qwen3-32b": "qwen/qwen3-32b",
    }
    FALLBACK_MODEL = "llama-3.1-8b-instant"

    def __init__(self, model: str = "mistral"):
        model_id = self.AVAILABLE_MODELS.get(model, self.FALLBACK_MODEL)
        if not GROQ_API_KEY:
            raise ValueError(
                "Missing GROQ_API_KEY. Set it in llmchatbot/.env as GROQ_API_KEY=<your_key>."
            )
        self.llm = ChatGroq(
            model=model_id,
            groq_api_key=GROQ_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        self.model_name = model_id

    def _build_messages(self, system_prompt: str, human_message: str):
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]

    def _is_decommissioned_error(self, exc: Exception) -> bool:
        if not isinstance(exc, BadRequestError):
            return False
        text = str(exc).lower()
        return "model_decommissioned" in text or "decommissioned" in text

    def generate(self, system_prompt: str, human_message: str) -> str:
        messages = self._build_messages(system_prompt, human_message)
        try:
            response = self.llm.invoke(messages)
        except Exception as e:
            # Groq retires model IDs periodically; auto-recover to a known live model.
            if self._is_decommissioned_error(e) and self.model_name != self.FALLBACK_MODEL:
                self.llm = ChatGroq(
                    model=self.FALLBACK_MODEL,
                    groq_api_key=GROQ_API_KEY,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                self.model_name = self.FALLBACK_MODEL
                response = self.llm.invoke(messages)
            else:
                raise
        return response.content

    def stream(self, system_prompt: str, human_message: str):
        """Stream tokens for real-time UI response."""
        messages = self._build_messages(system_prompt, human_message)
        try:
            for chunk in self.llm.stream(messages):
                yield chunk.content
            return
        except Exception as e:
            if self._is_decommissioned_error(e) and self.model_name != self.FALLBACK_MODEL:
                self.llm = ChatGroq(
                    model=self.FALLBACK_MODEL,
                    groq_api_key=GROQ_API_KEY,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                self.model_name = self.FALLBACK_MODEL
                for chunk in self.llm.stream(messages):
                    yield chunk.content
                return
            raise
