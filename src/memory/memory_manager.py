from src.memory.conversation_memory import ConversationMemory
from src.memory.entity_memory import EntityMemory
from src.memory.episodic_memory import EpisodicMemory
from src.memory.vector_memory import PineconeMemory
from src.memory.memory_gate import MemoryGate

class MemoryManager:


    def __init__(self,llm):

        self.conversation = ConversationMemory()
        self.entities = EntityMemory()
        self.episodic = EpisodicMemory()
        self.vector = PineconeMemory(llm=llm)
        self.memory_gate = MemoryGate(llm=llm)

    def build_context(self, query):

        history = self.conversation.get_history()

        entity_context = self.entities.get_entities()

        episodic_context = self.episodic.get_events()

        vector_context = self.vector.retrieve(query)

        context = f"""


                Conversation History:
                {history}

                Medical Entities:
                {entity_context}

                Past Important Events:
                {episodic_context}

                Relevant Past Interactions:
                {vector_context}
                """


        return context

    def store_interaction(self, user_query, response):

        # conversation
        self.conversation.save(user_query, response)

        # entity extraction
        self.entities.extract_entities(user_query)

        # episodic memory
        event = f"User asked: {user_query}"
        self.episodic.add_event(event)

        # vector memory
        self.vector.store(user_query, response)

