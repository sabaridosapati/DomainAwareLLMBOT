from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from config import OPENAI_API_KEY, TEMPERATURE, MAX_TOKENS


class OpenAIModel:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        self.model_name = model

    def generate(self, system_prompt: str, human_message: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ]
        response = self.llm.invoke(messages)
        return response.content

    def stream(self, system_prompt: str, human_message: str):
        """Stream tokens for real-time UI."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ]
        for chunk in self.llm.stream(messages):
            yield chunk.content