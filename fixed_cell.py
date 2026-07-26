import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import re
import uuid

def wrap_rtl(text):
    # Detect if text has Arabic characters
    if re.search("[؀-ۿ]", text):
        return f"<div dir='rtl' style='text-align: right;'>\n\n{text}\n\n</div>"
    return text

def process_interaction(audio_filepath, file_paths, text_input, history):
    uploaded_context = ""
    # 2. Process uploaded files dynamically within the query
    if file_paths:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        for fp in file_paths:
            docs = load_document(fp)
            if docs:
                for d in docs:
                    uploaded_context += f"\n[Uploaded Document]: {d.page_content}\n"
                chunks = text_splitter.split_documents(docs)
                if chunks:
                    pipeline.vector_store.add_documents(chunks)

    if audio_filepath:
        asr = EgyptianASR()
        user_input = asr.transcribe(audio_filepath)
    else:
        user_input = text_input

    # If the user uploads a document but doesn't type anything, set a default prompt
    if not user_input and uploaded_context:
        user_input = "I have uploaded a document. Please read it and answer any questions inside it, or summarize it if there are no questions."

    if not user_input:
        return "", None, None, history, None

    # We keep a display version of user_input for the UI
    display_user_input = user_input

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

    # Inject the uploaded text into chat history so the LLM sees it directly
    if uploaded_context:
        # Truncate to avoid context window explosion (e.g. max 15000 chars)
        if len(uploaded_context) > 15000:
            uploaded_context = uploaded_context[:15000] + "\n... (truncated due to length)"
        
        chat_history.append(
            SystemMessage(content=f"The user just uploaded a document. Here is its content:\n{uploaded_context}\n\nPlease use this content to answer the user's query.")
        )

    result = pipeline.query(user_input, chat_history=chat_history)
    answer = result["answer"]

    # Detect language of the answer and choose TTS voice
    is_arabic = bool(re.search("[؀-ۿ]", answer))
    voice = "ar-EG-SalmaNeural" if is_arabic else "en-US-AriaNeural"
    tts = HighQualityTTS(voice=voice)
    
    # Use a unique filename so Gradio triggers autoplay each time
    out_audio = f"response_{uuid.uuid4().hex}.mp3"

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

    # Add back the display version of user input to history
    history.append({"role": "user", "content": display_user_input})
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
