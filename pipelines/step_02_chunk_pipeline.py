from src.ingestion.chunker import DocumentChunker
from src.logger import logging


def main():
    try:
        logging.info("STEP 2: Starting document chunking pipeline")

        chunker = DocumentChunker()
        chunker.run_chunking()

        logging.info("STEP 2: Document chunking completed successfully")

    except Exception as e:
        logging.exception(f"STEP 2 FAILED: {e}")
        raise e


if __name__ == "__main__":
    main()