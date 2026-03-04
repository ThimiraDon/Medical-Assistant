import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer
from src.logger import logging
from src.config import MODEL_NAME,CHUNK_SIZE,OVERLAP_SIZE

from src.config import PROCESSED_DATA_PATH, CHUNKED_DATA_PATH

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class DocumentChunker:
    def __init__(self, input_path=PROCESSED_DATA_PATH, output_path=CHUNKED_DATA_PATH):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def load_cleaned_docs(self):
        try:
            logging.info(f"Loading cleaned documents from {self.input_path}")
            with open(self.input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = [Document(page_content=item["text"], metadata=item["metadata"]) for item in data]
            logging.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logging.exception(f"Failed to load cleaned documents: {e}")
            return []

    def chunk_documents(self, documents, chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE):
        try:
            
            splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
            )

            chunks = splitter.split_documents(documents)
            logging.info(f"Created {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logging.exception(f"Failed to chunk documents: {e}")
            return []

    def save_chunks(self, chunks):
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = [{"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]

            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)

            logging.info(f"Saved {len(chunks)} chunks to {self.output_path}")
        except Exception as e:
            logging.exception(f"Failed to save chunks: {e}")

    def run_chunking(self):
        docs= self.load_cleaned_docs()
        if not docs:
            logging.warning("No documents to chunk.")
            return  
        chunks = self.chunk_documents(docs)
        if not chunks:
            logging.warning("No chunks created.")
            return                  
        self.save_chunks(chunks)    
