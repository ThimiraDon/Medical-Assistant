import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone Index
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-assistant")

# Validation
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

if PINECONE_API_KEY is None:
    raise ValueError("PINECONE_API_KEY not set. Add it to your .env file.")