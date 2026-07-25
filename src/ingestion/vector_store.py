import os
from typing import List
import logging
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Using a smaller multilingual model to prevent OOM kills
# Note: It requires downloading a model (~470MB)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFace embeddings model.
    """
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        # Using CPU for now, can be configured for cuda or mps if available
        model_kwargs = {'device': 'cpu'} 
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise

def setup_vector_store(
    documents: List[Document], 
    persist_directory: str = "./data/chroma_db",
    collection_name: str = "te_knowledge_base"
) -> Chroma:
    """
    Splits documents into chunks, generates embeddings, and stores them in ChromaDB.
    """
    if not documents:
        logger.warning("No documents provided to vector store.")
        # If we just want to load the existing DB
        embeddings = get_embeddings_model()
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        return vector_store

    logger.info(f"Processing {len(documents)} documents for ingestion.")
    
    # Split documents into smaller chunks for better retrieval
    # For Arabic text, standard character splitting is usually fine, 
    # but we should ensure we don't break words mid-way.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")
    
    # Initialize embeddings
    embeddings = get_embeddings_model()
    
    # Create and persist the vector store
    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    logger.info(f"Storing chunks in ChromaDB at {persist_directory}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    logger.info("Vector store updated successfully.")
    return vector_store

def get_vector_store(
    persist_directory: str = "./data/chroma_db",
    collection_name: str = "te_knowledge_base"
) -> Chroma:
    """
    Returns an existing ChromaDB vector store instance.
    """
    embeddings = get_embeddings_model()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_store
