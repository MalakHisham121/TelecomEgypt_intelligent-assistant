<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/WE_logo.svg/512px-WE_logo.svg.png" alt="Telecom Egypt WE Logo" width="150"/>
  <h1>Telecom Egypt Intelligent Assistant</h1>
</div>

An advanced Retrieval-Augmented Generation (RAG) assistant designed for Telecom Egypt. The system seamlessly processes both text and voice inputs (including Egyptian Arabic), retrieves relevant organizational context, and provides intelligent, natural-sounding voice and text responses.

## ✨ Features
- **Multimodal Inputs & Outputs:** Supports text chat and direct microphone recordings.
- **Dialect-Aware ASR:** Employs `faster-whisper` for accurate Egyptian Arabic and English code-switching transcriptions.
- **Offline TTS:** Utilizes `piper-tts` for high-quality, localized Arabic text-to-speech generation.
- **Advanced RAG Architecture:** Integrates `LangChain`, `ChromaDB`, and `Ollama` for context-aware answering from uploaded internal documents.
- **Modern Interface:** A sleek, brand-aligned UI built with `Streamlit`.

## 🛠️ Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.
- Adequate RAM for local inference (8GB+ minimum).

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   # Clone your repo here
   cd TelecomEgypt_intelligent-assistant
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull the required local LLM (via Ollama):**
   ```bash
   ollama pull qwen2.5:0.5b
   ```

## 🎮 How to Run

1. Ensure the **Ollama** service is active on your machine.
2. Start the application from your terminal:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
4. Upload your PDFs or text documents via the **Knowledge Base** sidebar to train the assistant on your data.
5. Interact using the text chat or the built-in microphone widget!

---

### 📝 Developer Drafts & Notes
*(Kept for future documentation and presentation building)*

**Challenges I faced through building this project:**
1. First time to use ASR, TTS, OCR models
2. Web Scraping
3. Complex RAG architecture
4. Different input and different output capabilities needed
5. Arabic and Egyptian supported models

**Don't forget to:**
1. Make the ppt
2. Make readme (Done)
3. Diagram and workflow

**Why not use:**
1. Colab -> somewhat hard as notebooks cannot talk to each other easily; scripts are easier to manage and modularize.
2. More advanced models -> resource limitations and complexity (RAM, GPU), I want to run locally.