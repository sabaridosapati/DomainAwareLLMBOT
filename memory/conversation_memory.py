from collections import defaultdict
from langchain.memory import ConversationBufferWindowMemory


class ConversationMemoryManager:
    """
    Per-session, per-domain conversation memory.
    Keeps last N turns to stay within context window.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        # session_id → domain → memory
        self._memories = defaultdict(dict)

    def get_memory(self, session_id: str, domain: str) -> ConversationBufferWindowMemory:
        if domain not in self._memories[session_id]:
            self._memories[session_id][domain] = ConversationBufferWindowMemory(
                k=self.window_size,
                return_messages=True,
                memory_key="chat_history"
            )
        return self._memories[session_id][domain]

    def save_turn(self, session_id: str, domain: str,
                  human_msg: str, ai_msg: str):
        memory = self.get_memory(session_id, domain)
        memory.save_context(
            {"input": human_msg},
            {"output": ai_msg}
        )

    def get_history(self, session_id: str, domain: str) -> list:
        memory = self.get_memory(session_id, domain)
        return memory.load_memory_variables({}).get("chat_history", [])

    def clear(self, session_id: str):
        if session_id in self._memories:
            del self._memories[session_id]