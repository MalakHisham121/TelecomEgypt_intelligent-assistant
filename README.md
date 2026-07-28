
<div align="center">
  <h1>Telecom Egypt Intelligent Assistant</h1>
  <p><strong>A Multimodal RAG & Bilingual Voice Assistant Powered by Groq, LangChain, and Gradio</strong></p>
  <p>
    <a href="https://huggingface.co/spaces/MalakHisham/te-intelligent-assistant"><strong>Try the Live Demo on Hugging Face Spaces</strong></a>
  </p>
</div>

An advanced Retrieval-Augmented Generation (RAG) assistant designed for Telecom Egypt (WE). The system seamlessly processes text, natural voice audio, and scanned document images, retrieves relevant corporate knowledge from ChromaDB using multi-turn query rewriting, and delivers intelligent, natural-sounding voice and text responses in both Egyptian Arabic and English.

---

## Architecture Workflow

The following diagram illustrates the data flow, multimodal document ingestion, conversational query rewriting, and voice/text loop of the assistant:

```mermaid
graph TD
    User([User])

    subgraph UI [Frontend Interface]
        App[Gradio Web Interface<br/>Custom WE Theme & RTL Support]
    end

    subgraph Ingestion [Knowledge Base Setup]
        Loaders[Multimodal Loaders<br/>BeautifulSoup Web Scraper, PDF, DOCX, TXT]
        Chroma[(ChromaDB Vector Store<br/>paraphrase-multilingual-MiniLM-L12-v2)]
        Loaders -->|"Embed Chunks"| Chroma
    end

    subgraph Processing [Multimodal Engines]
        ASR[ASR: Whisper-large-v3-turbo]
        OCR[Vision OCR: Llama-3.2-90B-Vision-Instruct]
    end

    subgraph RAG [RAG & Generation Pipeline]
        Rewriter[Conversational Query Rewriter<br/>Llama-3.3-70B-Versatile]
        Retriever[Top-K MMR Semantic Retriever]
        LLM[LLM: Llama-3.3-70B-Versatile]
        TTS[TTS: Edge-TTS Neural Voices]
    end

    %% Input Routing
    User -->|"Audio / Images / Text"| App
    App -->|"Voice Audio"| ASR
    App -->|"Uploaded Images"| OCR
    App -->|"Uploaded Docs"| Loaders
    
    %% Processing Routing
    ASR -->|"Transcribed Text"| App
    OCR -->|"Extracted Text"| Loaders
    
    %% Retrieval & Generation
    App -->|"Raw Query + History"| Rewriter
    Rewriter -->|"Standalone Search Query"| Retriever
    Chroma -.->|"Relevant FAQ Chunks"| Retriever
    Retriever -->|"Formatted Docs + History"| LLM
    
    %% Output Routing
    LLM -->|"Streaming Text (<think> filtered)"| App
    LLM -->|"Clean Answer Text"| TTS
    TTS -->|"Synthesized MP3 Audio"| App
    App -->|"Voice Response & RTL Text"| User

```

---

## Key Features

* **Conversational Query Rewriting:** Eliminates multi-turn context loss. The pipeline evaluates conversation history and automatically translates contextual follow-ups (e.g., *"How much is it?"*) into standalone vector search queries (e.g., *"How much is WE Space?"*) before querying ChromaDB.
* **Ultra-Fast Multimodal RAG:** Powered by Groq's `llama-3.3-70b-versatile` for near-instantaneous reasoning, deep step-by-step thinking (`<think>` tag parsing), and accurate context mapping.
* **Bilingual Speech Engine:**
* **Speech-to-Text (ASR):** Leverages `whisper-large-v3-turbo` with domain-specific Egyptian telecom vocabulary prompts (*"This is a customer service conversation. هذه محادثة لخدمة العملاء باللهجة المصرية."*) to capture natural speech without accidental translation.
* **Text-to-Speech (TTS):** Utilizes `edge-tts` with dynamic voice switching for natural-sounding Arabic (`ar-EG-SalmaNeural`) and English (`en-US-AriaNeural`) audio responses.


* **Multimodal Document & Image OCR:**
* Native document loaders for PDF (`pypdf`), DOCX (`docx2txt`), and TXT files.
* Integrated Vision OCR (`llama-3.2-90b-vision-instruct`) to accurately extract bilingual text from scanned documents and images (`.png`, `.jpg`, `.jpeg`).


* **Multilingual Vector Embeddings:** Uses Hugging Face's `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for cross-lingual semantic matching.
* **Strict Language & RTL Mirroring:** Built-in system directives guarantee that English queries receive English responses and Arabic queries receive Arabic responses. The UI dynamically wraps Arabic outputs in Right-to-Left (`<div dir='rtl'>`) HTML containers.
* **Custom Telecom Egypt Branding:** Features a custom Gradio theme (`we_theme`) styled with Telecom Egypt's official purple/indigo color palette and embedded brand logos.

---

## How to Run the Application

You can access the live cloud deployment directly, run the self-contained notebook (`TelecomEgypt_Assistant.ipynb`) in Google Colab/Jupyter (where package installations are handled inside the notebook cells), or execute the standalone Python script.

### Option A: Live Demo (Hugging Face Spaces)

The fastest way to experience the assistant is through the live web deployment. No installation or API keys are required.

* **Live Deployment URL:** [MalakHisham/te-intelligent-assistant](https://huggingface.co/spaces/MalakHisham/te-intelligent-assistant)

---

### Option B: Running the Notebook in Google Colab (Zero Setup)

Running the notebook in Google Colab requires no manual installation steps. All dependencies are pre-configured in the notebook cells.

1. Open `TelecomEgypt_Assistant.ipynb` in **[Google Colab](https://colab.research.google.com/)**.
2. Run **Section 1** to execute the built-in package installation cell.
3. Run **Section 2** and securely paste your free `GROQ_API_KEY` when prompted.
4. Execute remaining cells sequentially. The notebook automatically scrapes WE FAQ pages and builds the local Chroma vector store.
5. In **Section 12**, the Gradio web interface launches inline directly inside the notebook alongside a temporary public share link (`https://xxxx.gradio.live`)!

---

### Option C: Running the Notebook Locally (Jupyter Notebook / Lab)

If executing the notebook locally, package installation is handled by running the notebook's first cell.

1. **Clone the repository and enter the directory:**
```bash
git clone <your-repo-url>
cd TelecomEgypt_intelligent-assistant

```


2. **Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Launch Jupyter:**
```bash
pip install jupyter
jupyter notebook

```


4. **Execute the Notebook:**
* Open `TelecomEgypt_Assistant.ipynb`.
* Run **Section 1** to automatically install all required packages into your active environment.
* Enter your `GROQ_API_KEY` in **Section 2**.
* Run all remaining cells sequentially. Section 12 will launch the interactive Gradio UI locally in your browser!



---
