from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re,os
import json

from src.ingestion import STARTING_PAGE_NUMBER, RAWDATA_PATH,PROCESSED_DATA_PATH



def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        show_progress=True,
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    docs = [doc for doc in documents if doc.metadata.get('page', 0) >= STARTING_PAGE_NUMBER]
    return docs

def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text:
    - Removes multiple newlines
    - Removes extra spaces
    - Optionally remove headers/footers
    """
    # Remove common headers (adjust pattern if needed)
    text = re.sub(r'GALE ENCYCLOPEDIA.*', '', text)
    
    # Remove common footers with page numbers
    text = re.sub(r'GEM - .*Page \d+', '', text)
    
    # Remove standalone page numbers like "Page 69"
    text = re.sub(r'Page \d+', '', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    return text

def clean_documents(documents):
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    return documents

import json

def save_cleaned_docs(docs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    serializable_docs = []

    for doc in docs:
        serializable_docs.append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)

def load_and_clean_data():
    documents = load_pdf_file(RAWDATA_PATH)
    cleaned_docs = clean_documents(documents)
    save_cleaned_docs(cleaned_docs, PROCESSED_DATA_PATH)
