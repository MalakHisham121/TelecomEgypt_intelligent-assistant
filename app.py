import streamlit as st
import os

# Import UI components and dependencies
from src.ui.styles import apply_theme
from src.ui.sidebar import render_sidebar
from src.ui.chat import render_chat_interface
from src.utils.dependencies import load_models

def main():
    # 1. Apply global configuration and styling
    apply_theme()
    
    # 2. Load and cache AI models
    rag_pipeline, asr_model, tts_model = load_models()
    
    # 3. Ensure required directories exist
    os.makedirs("tmp", exist_ok=True)
    os.makedirs("data/chroma_db", exist_ok=True)
    
    # 4. Render Sidebar for Document Uploading (Batch Question Mode)
    render_sidebar(rag_pipeline)
    
    # 5. Render Main Chat Interface
    render_chat_interface(rag_pipeline, asr_model, tts_model)

if __name__ == "__main__":
    main()
