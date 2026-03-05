from src.embeddings.pinecone_loader import PineconeLoader
from src.logger import logging


def main():
    try:
        logging.info("STEP 3: Starting embeddings and Pinecone upload pipeline")

        pinecone_loader = PineconeLoader()
        pinecone_loader.load_embeddings_to_pinecone()

        logging.info("STEP 3: Embeddings uploaded to Pinecone successfully")

    except Exception as e:
        logging.exception(f"STEP 3 FAILED: {e}")
        raise e


if __name__ == "__main__":
    main()