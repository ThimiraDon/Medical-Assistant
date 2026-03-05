from src.ingestion.loader import PDFProcessor
from src.logger import logging


def main():
    try:
        logging.info("STEP 1: Starting PDF loading and cleaning pipeline")

        pdf_loader = PDFProcessor()
        pdf_loader.load_and_clean_data()

        logging.info("STEP 1: PDF cleaning completed successfully")

    except Exception as e:
        logging.exception(f"STEP 1 FAILED: {e}")
        raise e


if __name__ == "__main__":
    main()