import streamlit as st
from src.llm import RAGPipeline
from src.speech import EgyptianASR, OfflineTTS

@st.cache_resource
def load_models():
    """Load and cache the models so they don't reload on every interaction."""
    # RAG pipeline setup
    rag = RAGPipeline(model_name="qwen2.5:0.5b")
    # ASR setup (using tiny to conserve RAM)
    asr = EgyptianASR(model_size="tiny")
    # TTS setup
    tts = OfflineTTS(voice="ar_JO-kareem-low")
    return rag, asr, tts
