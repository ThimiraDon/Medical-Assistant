from pathlib import Path
import os

STARTING_PAGE_NUMBER = 14

raw_path = r"C:\Users\Thimira\Desktop\Data\ML & AI\Deep Learning\LangChain\Medical-Assistant\Data\raw"
RAWDATA_PATH = Path(raw_path)

CLEANED_DATA_FILENAME = "cleaned_docs.json"
cleaned_path=r"data/processed/"
PROCESSED_DATA_PATH = Path(os.path.join(cleaned_path, CLEANED_DATA_FILENAME))

CHUNKED_DATA_FILENAME = "chunks.json"
CHUNKED_DATA_PATH = Path(os.path.join(cleaned_path, CHUNKED_DATA_FILENAME))

#Embedding model name
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE=500
OVERLAP_SIZE=50

#Pinecone index name
INDEX_NAME = "medical-assistant"

#retriever
TOP_K = 5
SEARCH_TYPE = "similarity"

#llm
LLM_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.0

#Memory
MEMORY_INDEX="medical-assistant-memory"