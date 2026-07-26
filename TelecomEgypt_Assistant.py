!pip install langchain langchain-community langchain-huggingface langchain-chroma chromadb sentence-transformers transformers edge-tts pypdf python-docx docx2txt pillow beautifulsoup4 requests langchain-groq groq

!pip install docx2txt

import os
import requests
from bs4 import BeautifulSoup
from typing import List
import logging
from PIL import Image
from getpass import getpass

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("Please enter your Groq API Key:")
os.environ["GROQ_API_KEY"] = getpass()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAG_SYSTEM_PROMPT = """You are a helpful and intelligent assistant for Telecom Egypt (WE).
Your primary task is to answer the user's question based on the provided context and the conversation history.

Follow these STRICT rules:
1. ONLY use the provided context to answer the question. If the answer is not in the context, say "I do not have enough information to answer that based on the provided context." Do not use your own external knowledge.
2. CRITICAL: YOU MUST DETECT THE USER'S LANGUAGE AND REPLY IN THE SAME LANGUAGE. If the user asks in English, you MUST reply in English. If the user asks in Arabic or Egyptian Arabic, you MUST reply in Arabic. Do not answer in English if the question is in Arabic!
3. DO NOT include source citations inside your response. Provide a natural and continuous answer.

Context:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

import base64
from groq import Groq
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def process_image_ocr(file_path: str) -> List[Document]:
    try:
        with open(file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        client = Groq()
        logger.info(f"Running high-quality OCR via Groq Vision for {file_path}...")
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all the text from this image exactly as written. Ensure Arabic and English text is captured perfectly. Output ONLY the extracted text, with absolutely no conversational filler or commentary."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_completion_tokens=2048,
        )
        text = completion.choices[0].message.content
        return [Document(page_content=text, metadata={"source": file_path, "type": "image_ocr"})]
    except Exception as e:
        logger.error(f"Error processing image {file_path}: {e}")
        return []

def load_document(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf': return PyPDFLoader(file_path).load()
        elif ext == '.docx': return Docx2txtLoader(file_path).load()
        elif ext == '.txt': return TextLoader(file_path, encoding='utf-8').load()
        elif ext in ['.png', '.jpg', '.jpeg']: return process_image_ocr(file_path)
        else:
            logger.warning(f"Unsupported file extension: {ext}")
            return []
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

def scrape_te_page(url: str) -> List[Document]:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.decompose()

        text = soup.get_text(separator=' ', strip=True)
        logger.info(f"Successfully scraped {len(text)} characters from {url}")
        return [Document(page_content=text, metadata={"source": url, "type": "web_page"})]
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return []


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embeddings_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def setup_vector_store(documents: List[Document], persist_directory: str = "./data/chroma_db") -> Chroma:
    if not documents:
        return Chroma(collection_name="te_knowledge_base", embedding_function=get_embeddings_model(), persist_directory=persist_directory)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")

    os.makedirs(persist_directory, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings_model(),
        persist_directory=persist_directory,
        collection_name="te_knowledge_base"
    )

# List of high-value TE knowledge targets
faq_urls = [
    "https://www.te.eg/about-te/faq",                  # Main Mobile & USSD FAQs
    "https://te.eg/en/about-te/faq/fixed-broadband",   # Home Internet (WE Space, Routers, Quotas)
    "https://te.eg/en/about-te/faq/fixed-voice"        # Landline (Billing, Installments, Tariffs)
]

all_docs = []
for url in faq_urls:
    logger.info(f"Processing knowledge target: {url}")
    # Using our updated Jina AI scraper from Section 4
    docs = scrape_te_page(url)
    all_docs.extend(docs)

# Set up the Chroma DB and generate embeddings from all scraped FAQ pages
vector_db = setup_vector_store(all_docs, persist_directory="./data/chroma_db")
logger.info(f"Knowledge base successfully populated with {len(all_docs)} source pages!")

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'Unknown')}]\nContent: {d.page_content}" for d in docs)

class GroqRAGPipeline:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.vector_store = Chroma(
            collection_name="te_knowledge_base",
            embedding_function=get_embeddings_model(),
            persist_directory="./data/chroma_db"
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.llm = ChatGroq(model_name=model_name, temperature=0.1)

        self.rag_chain = (
            {
                "context": lambda x: format_docs(self.retriever.invoke(x["input"])),
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"]
            }
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, user_input: str, chat_history: list = None) -> dict:
        docs = self.retriever.invoke(user_input)
        answer = self.rag_chain.invoke({"input": user_input, "chat_history": chat_history or []})
        return {"answer": answer, "context": docs}

logger.info("Initializing Groq RAG Pipeline...")
pipeline = GroqRAGPipeline(model_name="llama-3.3-70b-versatile")
logger.info("Pipeline Ready!")

query = "What services does Telecom Egypt offer for personal use?"
result = pipeline.query(query)

print("\nTE Assistant:")
print("-" * 60)
print(result["answer"])
print("-" * 60)

if result["context"]:
    print("\nSources:")
    sources = set(doc.metadata.get('source', 'Unknown') for doc in result["context"])
    for source in sources:
        print(f"  • {source}")

import subprocess
import logging
from groq import Groq

logger = logging.getLogger(__name__)

class EgyptianASR:
    def __init__(self):
        self.client = Groq()

    def transcribe(self, audio_path: str) -> str:
        logger.info(f"Transcribing {audio_path} instantly with Groq Whisper...")
        with open(audio_path, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                prompt="هذه محادثة لخدمة العملاء باللهجة المصرية.",
                response_format="json",
                language="ar"
            )
        return transcription.text

class HighQualityTTS:
    def __init__(self, voice="ar-EG-SalmaNeural"):
        self.voice = voice

    def synthesize(self, text: str, output_path: str):
        logger.info(f"Synthesizing text using Edge TTS ({self.voice})")
        # Run edge-tts via command line to avoid ALL async/await bugs in Colab!
        subprocess.run(
            ["edge-tts", "--voice", self.voice, "--text", text, "--write-media", output_path],
            check=True
        )


from IPython.display import Audio

try:
    # Optional: Test ASR (Requires an uploaded audio file like "test_audio.wav")
    # asr = EgyptianASR()
    # transcript = asr.transcribe("test_audio.wav")
    # print(transcript)

    # Initialize TTS and synthesize the result
    tts = HighQualityTTS(voice="ar-EG-SalmaNeural")
    audio_file = "response_output.mp3"

    # Generate audio (using Colab's existing event loop)
    tts.synthesize(result["answer"], audio_file)

    logger.info("Generated audio successfully.")
except Exception as e:
    logger.error(f"Audio processing failed: {e}")

# Audio(audio_file) # Uncomment to play in notebook


!pip install gradio


!pip install --upgrade gradio

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
import re

def wrap_rtl(text):
    # Detect if text has Arabic characters
    if re.search("[؀-ۿ]", text):
        return f"<div dir='rtl' style='text-align: right;'>\n\n{text}\n\n</div>"
    return text

def process_interaction(audio_filepath, file_paths, text_input, history):
    # 2. Process uploaded files dynamically within the query
    if file_paths:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        for fp in file_paths:
            docs = load_document(fp)
            if docs:
                chunks = text_splitter.split_documents(docs)
                if chunks:
                    pipeline.vector_store.add_documents(chunks)

    if audio_filepath:
        asr = EgyptianASR()
        user_input = asr.transcribe(audio_filepath)
    else:
        user_input = text_input

    if not user_input:
        return "", None, None, history, None

    # 1. Parse modern Gradio dictionary format into LangChain messages
    chat_history = []
    for msg in history:
        content = msg["content"]
        if isinstance(content, list) or isinstance(content, tuple):
            content = content[0] if len(content) > 0 else ""
        content = str(content)
        
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=content))
        elif msg["role"] == "assistant":
            clean_content = content.replace("<div dir='rtl' style='text-align: right;'>\n\n", "").replace("\n\n</div>", "")
            chat_history.append(AIMessage(content=clean_content))

    result = pipeline.query(user_input, chat_history=chat_history)
    answer = result["answer"]

    tts = HighQualityTTS(voice="ar-EG-SalmaNeural")
    out_audio = "response.mp3"

    try:
        tts.synthesize(answer, out_audio)
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        out_audio = None

    # Append sources at the bottom
    answer_text = answer
    if result.get("context"):
        sources = set(doc.metadata.get('source', 'Unknown') for doc in result["context"])
        if sources:
            answer_text += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
            
    # 3. Apply RTL if Arabic
    answer_text_formatted = wrap_rtl(answer_text)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": answer_text_formatted})

    return "", None, None, history, out_audio

with gr.Blocks(css=".gradio-container {max-width: 900px; margin: auto;}") as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'>Telecom Egypt Intelligent Assistant</h1>"
    )
    chatbot = gr.Chatbot(label="TE Assistant", height=500)
    
    # 5. Make audio_output visible so the user can play it manually if autoplay is blocked
    audio_output = gr.Audio(
        label="Assistant Voice Response", autoplay=True, visible=True
    )
    
    with gr.Row():
        with gr.Column(scale=8):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Type your message here...",
                container=False
            )
        with gr.Column(scale=1, min_width=80):
            # 1. Send button explicitly
            submit_btn = gr.Button("Send", variant="primary")
            
    with gr.Row():
        audio_in = gr.Audio(
            sources=["microphone"], type="filepath", label="Record Voice (Optional)"
        )
        file_in = gr.File(label="Attach Documents (Optional)", file_count="multiple")

    submit_btn.click(
        process_interaction,
        inputs=[audio_in, file_in, txt, chatbot],
        outputs=[txt, audio_in, file_in, chatbot, audio_output],
    )
    txt.submit(
        process_interaction,
        inputs=[audio_in, file_in, txt, chatbot],
        outputs=[txt, audio_in, file_in, chatbot, audio_output],
    )

# Launch cleanly
demo.launch(share=True, debug=True)


