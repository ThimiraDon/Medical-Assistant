from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import MODEL_NAME, INDEX_NAME, TOP_K
from pinecone import Pinecone

from src.config.API_config import PINECONE_API_KEY


class MultiQueryRetriever:
    """
    Retriever that supports multi-query input and query rewriting
    with similarity threshold filtering.
    """

    def __init__(self, llm, memory_manager, threshold: float = 0.65):

        self.embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.threshold = threshold

        if INDEX_NAME not in [idx.name for idx in self.pc.list_indexes()]:
            raise ValueError(f"Pinecone index {INDEX_NAME} does not exist.")

        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=self.embedding
        )

    def retrieve(self, queries):

        all_docs = []

        for q in queries:

            # retrieve with scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                q,
                k=TOP_K
            )

            for doc, score in docs_with_scores:

                # Apply threshold filter
                if score >= self.threshold:
                    all_docs.append(doc)

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