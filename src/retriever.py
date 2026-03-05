# src/retriever/retriever.py
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

from src.config import MODEL_NAME, INDEX_NAME, TOP_K,SEARCH_TYPE

from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set. Add it to your .env file.")

pc = Pinecone(api_key=PINECONE_API_KEY)

class MedicalRetriever:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        self.k = TOP_K
        self.search_type= SEARCH_TYPE
        self.index_name = INDEX_NAME

    def get_vectors(self):

        if self.index_name not in [idx.name for idx in pc.list_indexes()]:
            raise ValueError(f"Pinecone index {self.index_name} does not exist.")

        docsearch = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embedding
        )

        retriever = docsearch.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.k}
        )
        return retriever