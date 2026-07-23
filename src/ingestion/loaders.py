import os
from typing import List
import logging
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

def process_image_ocr(file_path: str) -> List[Document]:
    """
    Extracts text from an image using Tesseract OCR, configured for Arabic.
    """
    try:
        # Open the image file
        img = Image.open(file_path)
        
        # Configure tesseract for Arabic ('ara') and English ('eng')
        # Requires tesseract-ocr and tesseract-ocr-ara to be installed on the system
        try:
            text = pytesseract.image_to_string(img, lang='ara+eng')
            
            doc = Document(
                page_content=text,
                metadata={"source": file_path, "type": "image_ocr"}
            )
            return [doc]
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not in your PATH. Image OCR failed.")
            return []
            
    except Exception as e:
        logger.error(f"Error processing image {file_path}: {e}")
        return []

def load_document(file_path: str) -> List[Document]:
    """
    Loads a document based on its file extension.
    Supported extensions: .pdf, .docx, .txt, .png, .jpg, .jpeg
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
        
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
            return loader.load()
        elif ext == '.txt':
            # Specify utf-8 encoding for Arabic text files
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        elif ext in ['.png', '.jpg', '.jpeg']:
            return process_image_ocr(file_path)
        else:
            logger.warning(f"Unsupported file extension: {ext} for file {file_path}")
            return []
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        return []
