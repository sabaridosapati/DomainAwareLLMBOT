from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from config import GOOGLE_API_KEY, TEMPERATURE, MAX_TOKENS


class GeminiModel:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GOOGLE_API_KEY,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            convert_system_message_to_human=True
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
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ]
        for chunk in self.llm.stream(messages):
            yield chunk.content