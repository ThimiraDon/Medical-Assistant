from src.ingestion.loader import PDFProcessor
from src.ingestion.chunker import DocumentChunker
from src.embeddings.pinecone_loader import PineconeLoader

if __name__ == "__main__":
    # Step 1: Load and clean data
    pdf_loader = PDFProcessor()
    pdf_loader.load_and_clean_data()

    # Step 2: Run chunking
    doc_chunker = DocumentChunker()
    doc_chunker.run_chunking()

    # Step 3: Initialize Pinecone and upload chunks
    pinecone_loader = PineconeLoader()
    pinecone_loader.load_embeddings_to_pinecone()
