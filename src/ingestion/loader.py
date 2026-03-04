import os
import re
import json
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from src.config import STARTING_PAGE_NUMBER, RAWDATA_PATH, PROCESSED_DATA_PATH
from src.logger import logging  # Import the shared logger

class PDFProcessor:
    """
    A class to load, clean, and save PDF documents with logging.
    """
    def __init__(self, raw_data_path=RAWDATA_PATH, processed_data_path=PROCESSED_DATA_PATH,
                 starting_page=STARTING_PAGE_NUMBER):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.starting_page = starting_page

    def load_pdf_files(self):
        try:
            logging.info(f"Loading PDFs from {self.raw_data_path}")
            loader = DirectoryLoader(
                self.raw_data_path,
                glob="*.pdf",
                show_progress=True,
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            docs = [doc for doc in documents if doc.metadata.get('page', 0) >= self.starting_page]
            logging.info(f"Loaded {len(docs)} documents after filtering pages >= {self.starting_page}")
            return docs
        except Exception as e:
            logging.exception(f"Failed to load PDF files: {e}")
            return []

    @staticmethod
    def clean_text(text: str) -> str:
        try:
            text = re.sub(r'GALE ENCYCLOPEDIA.*', '', text)
            text = re.sub(r'GEM - .*Page \d+', '', text)
            text = re.sub(r'Page \d+', '', text)
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception as e:
            logging.exception(f"Failed to clean text: {e}")
            return text

    def clean_documents(self, documents):
        try:
            for doc in documents:
                doc.page_content = self.clean_text(doc.page_content)
            logging.info(f"Cleaned {len(documents)} documents")
            return documents
        except Exception as e:
            logging.exception(f"Failed to clean documents: {e}")
            return documents

    def save_cleaned_docs(self, docs):
        try:
            os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
            serializable_docs = [{
                "text": doc.page_content,
                "metadata": doc.metadata
            } for doc in docs]

            with open(self.processed_data_path, "w", encoding="utf-8") as f:
                json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved cleaned documents to {self.processed_data_path}")
        except Exception as e:
            logging.exception(f"Failed to save cleaned documents: {e}")

    def load_and_clean_data(self):
        try:
            documents = self.load_pdf_files()
            if not documents:
                logging.warning("No documents loaded to process.")
                return []

            cleaned_docs = self.clean_documents(documents)
            self.save_cleaned_docs(cleaned_docs)
            logging.info("PDF processing completed successfully.")
            return cleaned_docs
        except Exception as e:
            logging.exception(f"Error in load_and_clean_data workflow: {e}")
            return []