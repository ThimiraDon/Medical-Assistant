from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from pinecone import Pinecone, ServerlessSpec

from src.config import MODEL_NAME, MEMORY_INDEX
from src.config.API_config import PINECONE_API_KEY
from src.memory.memory_gate import MemoryGate

class PineconeMemory:
    """
    Long-term semantic memory using Pinecone.
    Stores important conversation interactions.
    """

    def __init__(self,llm=None, k: int = 3):
        self.llm = llm
        self.memory_gate = MemoryGate(llm)
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME
        )

        # Pinecone client
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # Detect embedding dimension
        embedding_dim = len(self.embeddings.embed_query("test"))

        # Create index if not exists
        if MEMORY_INDEX not in [idx.name for idx in self.pc.list_indexes()]:
            print(f"Creating memory index: {MEMORY_INDEX}")

            self.pc.create_index(
                name=MEMORY_INDEX,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        # Connect vector store
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=MEMORY_INDEX,
            embedding=self.embeddings
        )

        # Retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        # Memory wrapper
        self.memory = VectorStoreRetrieverMemory(
            retriever=retriever
        )
        self.index = self.pc.Index(MEMORY_INDEX)

    def retrieve(self, query: str) -> str:
        """
        Retrieve similar past memories.
        """
        memory = self.memory.load_memory_variables(
            {"prompt": query}
        )
        return memory.get("history", "")

    def store(self, user_query: str, response: str):
        """
        Save interaction to vector memory.
        """
        if self.memory_gate.should_store(user_query, response):
            self.memory.save_context(
                {"input": user_query},
                {"output": response}
            )

    def clear(self):
        try:
            self.index.delete(delete_all=True)
            print("Vector memory cleared.")
        except Exception as e:
            print("Vector reset failed:", e)
