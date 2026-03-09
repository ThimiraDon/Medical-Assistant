class MemoryGate:
    """
    Decides if an interaction contains important long-term information,
    fully using an LLM instead of hardcoded keywords.
    """

    def __init__(self, llm):
        """
        llm: Any LLM wrapper you have, e.g., your LLMLoader output.
        """
        self.llm = llm

    def should_store(self, user_query: str, response: str) -> bool:
        """
        Ask the LLM whether this interaction should be stored long-term.
        Returns True if the LLM says STORE, else False.
        """
        prompt = f"""
You are a memory manager for a personal medical assistant.

Decide if this interaction contains important long-term information
about the user. Consider personal information, health conditions,
preferences, allergies, or anything that the assistant should remember long-term.

Respond only with STORE or IGNORE.

User query: "{user_query}"
Assistant response: "{response}"
"""
        # Call the LLM
        decision = self.llm.invoke(prompt).content.strip().upper()
        return decision == "STORE"