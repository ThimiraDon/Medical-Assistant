from langchain.embeddings import HuggingFaceEmbeddings
from src.config import MODEL_NAME, INDEX_NAME, CHUNKED_DATA_PATH
from src.logger import logging

from pinecone import ServerlessSpec 
import json
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from pinecone import ServerlessSpec
from pinecone import Pinecone


class PineconeLoader:
    def __init__(self, model_name=MODEL_NAME, index_name=INDEX_NAME, chunks_path=CHUNKED_DATA_PATH):
        self.model_name = model_name
        self.index_name = index_name
        self.chunks_path = chunks_path

        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("PINECONE_API_KEY")

        # Initialize Pinecone client
        try:
            self.client = Pinecone(api_key=self.api_key)
            logging.info("Successfully initialized Pinecone client.")
        except Exception as e:
            logging.exception("Error initializing Pinecone client")
            raise e

    def load_chunks_from_json(self):
        """
        Load JSON chunks and convert to LangChain documents.
        """
        try:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            documents = [Document(page_content=item["text"], metadata=item["metadata"]) for item in data]
            logging.info(f"Loaded {len(documents)} chunks from {self.chunks_path}")
            return documents
        except Exception as e:
            logging.exception(f"Failed to load chunks from JSON: {e}")
            return []

    def download_embeddings(self):
        """
        Initialize the HuggingFace embeddings model.
        """
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            logging.info(f"Loaded HuggingFace embeddings: {self.model_name}")
            return self.embeddings
        except Exception as e:
            logging.exception(f"Failed to load embeddings model {self.model_name}: {e}")
            return None

    def create_pinecone_index(self):
        """
        Create Pinecone index if it does not exist.
        """
        try:
            existing_indexes = [idx.name for idx in self.client.list_indexes()]
            if self.index_name not in existing_indexes:
                self.client.create_index(
                    name=self.index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                logging.info(f"Created index {self.index_name}")
            else:
                logging.info(f"Index {self.index_name} already exists.")

            self.index = self.client.Index(self.index_name)
            return self.index
        except Exception as e:
            logging.exception(f"Failed to create or access Pinecone index {self.index_name}: {e}")
            return None

    def load_embeddings_to_pinecone(self):
        """
        Load JSON chunks, embed them, and upsert into Pinecone index.
        """
        try:
            documents = self.load_chunks_from_json()
            if not documents:
                logging.warning("No documents to upload to Pinecone.")
                return None

            if not hasattr(self, "embeddings"):
                self.download_embeddings()
            if not hasattr(self, "index"):
                self.create_pinecone_index()

            logging.info(f"Uploading {len(documents)} documents to Pinecone index '{self.index_name}'...")
            vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name,
            )
            logging.info(f"Successfully uploaded {len(documents)} chunks to Pinecone index '{self.index_name}'")
            return vectorstore

        except Exception as e:
            logging.exception("Failed to upload embeddings to Pinecone.")
            return None

        