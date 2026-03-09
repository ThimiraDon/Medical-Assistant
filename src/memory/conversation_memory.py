from langchain.memory import ConversationBufferWindowMemory
"""
Keeps recent conversational context
Prevents token explosion
Perfect for query rewriting
"""

class ConversationMemory:

    ##Short-term conversational memory.
    def __init__(self, k: int = 5):
        self.memory = ConversationBufferWindowMemory(
            k=k,
            return_messages=True
        )

    def get_history(self) -> str:
        """
        Returns formatted conversation history.
        """
        data = self.memory.load_memory_variables({})
        history = data.get("history", "")
        return history

    def save(self, user_query: str, response: str):
        """
        Save conversation turn.
        """
        self.memory.save_context(
            {"input": user_query},
            {"output": response}
        )

    def clear(self):
        """
        Reset memory.
        """
        self.memory.clear()
    
