import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer

from src.ingestion import PROCESSED_DATA_PATH, CHUNKED_DATA_PATH

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def load_cleaned_docs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    #converting json to langchain documents. because the chunking function takes 
    #       in langchain documents as input
    documents = [
        Document(page_content=item["text"], metadata=item["metadata"])
        for item in data
    ]

    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        #Ensures chunk fits embedding model input limit
        tokenizer=tokenizer, #make sure tokenizer follow its way to split text,not just split by character count
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap #kkep the semetic meaning of the text by overlapping chunks
    )
    return splitter.split_documents(documents)

def save_chunks(chunks, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = [
        {
            "text": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    
def run_chunking():
    documents = load_cleaned_docs(PROCESSED_DATA_PATH)
    chunks = chunk_documents(documents)

    save_chunks(
        chunks,
        CHUNKED_DATA_PATH
    )

    print(f"Total chunks created: {len(chunks)}")