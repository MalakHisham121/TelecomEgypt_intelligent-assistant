import os
import argparse
import logging
from src.ingestion.scraper import scrape_te_page
from src.ingestion.loaders import load_document
from src.ingestion.vector_store import setup_vector_store
from langchain_core.documents import Document


# Pipeline Maestro , takes URL and do scrapping , chunking , embedding then store work

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_url(url: str, persist_directory: str):
    logger.info(f"Starting ingestion for URL: {url}")
    docs = scrape_te_page(url)
    if docs:
        setup_vector_store(docs, persist_directory=persist_directory)
        logger.info(f"Successfully ingested {url}")
    else:
        logger.warning(f"No documents extracted from {url}")

def ingest_directory(dir_path: str, persist_directory: str):
    logger.info(f"Starting ingestion for directory: {dir_path}")
    all_docs = []
    if not os.path.isdir(dir_path):
        logger.error(f"Directory not found: {dir_path}")
        return
        
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            logger.info(f"Loading {file_path}...")
            docs = load_document(file_path)
            all_docs.extend(docs)
            
    if all_docs:
        setup_vector_store(all_docs, persist_directory=persist_directory)
        logger.info(f"Successfully ingested {len(all_docs)} documents from {dir_path}")
    else:
        logger.warning(f"No documents extracted from {dir_path}")

def main():
    parser = argparse.ArgumentParser(description="Ingest documents and URLs into the TE knowledge base.")
    parser.add_argument('--url', type=str, help='URL to scrape and ingest')
    parser.add_argument('--dir', type=str, help='Directory containing files to ingest')
    parser.add_argument('--db-dir', type=str, default='./data/chroma_db', help='ChromaDB persist directory')
    
    args = parser.parse_args()
    
    if not args.url and not args.dir:
        logger.error("Please provide either --url or --dir to ingest.")
        parser.print_help()
        return
        
    if args.url:
        ingest_url(args.url, args.db_dir)
        
    if args.dir:
        ingest_directory(args.dir, args.db_dir)

if __name__ == "__main__":
    main()
