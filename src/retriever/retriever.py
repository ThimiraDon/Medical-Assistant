from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import MODEL_NAME, INDEX_NAME, TOP_K, SEARCH_TYPE
from pinecone import Pinecone

from src.config.API_config import PINECONE_API_KEY
class MultiQueryRetriever:
    """
    Retriever that supports multi-query input and query rewriting
    """

    def __init__(self, llm,memory_manager):
        # Embeddings + Pinecone setup
        self.embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        if INDEX_NAME not in [idx.name for idx in self.pc.list_indexes()]:
            raise ValueError(f"Pinecone index {INDEX_NAME} does not exist.")

        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=self.embedding
        )

        self.retriever = vectorstore.as_retriever(
            search_type=SEARCH_TYPE,
            search_kwargs={"k": TOP_K}
        )

    def retrieve(self, queries):

        all_docs = []

        for q in queries:
            docs = self.retriever.invoke(q)
            all_docs.extend(docs)

        unique_docs = self.deduplicate_docs(all_docs)

        return unique_docs

    @staticmethod
    def deduplicate_docs(docs):
        """
        Remove duplicate document chunks based on page_content
        """
        seen = set()
        unique_docs = []
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                unique_docs.append(d)
        return unique_docs