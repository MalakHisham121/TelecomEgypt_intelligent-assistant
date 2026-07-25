import streamlit as st
import os
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from src.ui.styles import LOGO_URL
from src.utils.evaluation import Evaluator

def render_chat_interface(rag_pipeline, asr_model, tts_model):
    """
    Renders the main chat interface, processes text and audio inputs, 
    and handles state management for messages.
    """
    st.image(LOGO_URL, width=150)
    st.title("Telecom Egypt Intelligent Assistant")
    st.markdown("Welcome! You can interact via text or voice (using the microphone widget).")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "audio" in msg and os.path.exists(msg["audio"]):
                st.audio(msg["audio"], format="audio/wav")

    # Input Widgets
    user_audio = st.audio_input("Record a voice message")
    user_text = st.chat_input("Type your message here...")

    # Conditional Routing
    if user_text:
        _handle_text_input(user_text, rag_pipeline)
    elif user_audio:
        _handle_voice_input(user_audio, rag_pipeline, asr_model, tts_model)

def _get_chat_history():
    """Converts Streamlit session state into LangChain message objects."""
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            content = msg["content"].replace("(Voice) ", "")
            history.append(HumanMessage(content=content))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

def _handle_text_input(user_text, rag_pipeline):
    """Processes text input directly through the RAG pipeline."""
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Process via RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history = _get_chat_history()
            result = rag_pipeline.query(user_text, chat_history=history)
            answer = result["answer"]
            
            st.markdown(answer)
            _display_sources(result)
            
        with st.spinner("Evaluating response..."):
            evaluator = Evaluator()
            context_str = "\\n".join([doc.page_content for doc in result.get("context", [])])
            eval_scores = evaluator.evaluate(user_text, context_str, answer)
            _display_evaluation(eval_scores)
                        
    # Append assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})

def _handle_voice_input(user_audio, rag_pipeline, asr_model, tts_model):
    """Processes audio input through ASR, RAG, and TTS pipelines."""
    # Save audio input temporarily
    in_audio_path = os.path.join("tmp", f"input_{uuid.uuid4().hex}.wav")
    with open(in_audio_path, "wb") as f:
        f.write(user_audio.getbuffer())
        
    # Process ASR
    with st.chat_message("user"):
        with st.spinner("Transcribing..."):
            transcription = asr_model.transcribe(in_audio_path)
            
        display_text = f"(Voice) {transcription}"
        st.markdown(display_text)
        st.audio(in_audio_path, format="audio/wav")
    
    st.session_state.messages.append({
        "role": "user", 
        "content": display_text,
        "audio": in_audio_path
    })
    
    # Process via RAG & TTS
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history = _get_chat_history()
            result = rag_pipeline.query(transcription, chat_history=history)
            answer = result["answer"]
            
        with st.spinner("Synthesizing voice..."):
            out_audio_path = os.path.join("tmp", f"output_{uuid.uuid4().hex}.wav")
            try:
                tts_model.synthesize(answer, out_audio_path)
            except Exception as e:
                st.error(f"TTS Error: {e}")
                out_audio_path = None
        
        # Display Text and Audio
        st.markdown(answer)
        if out_audio_path and os.path.exists(out_audio_path):
            st.audio(out_audio_path, format="audio/wav")
            
        _display_sources(result)
        
        with st.spinner("Evaluating response..."):
            evaluator = Evaluator()
            context_str = "\\n".join([doc.page_content for doc in result.get("context", [])])
            eval_scores = evaluator.evaluate(transcription, context_str, answer)
            _display_evaluation(eval_scores)

    # Append assistant response to history
    history_msg = {"role": "assistant", "content": answer}
    if out_audio_path:
        history_msg["audio"] = out_audio_path
    st.session_state.messages.append(history_msg)

def _display_sources(result):
    """Helper to display document sources from the RAG response."""
    if result.get("context"):
        with st.expander("Sources"):
            sources = set(doc.metadata.get('source', 'Unknown') for doc in result["context"])
            for source in sources:
                st.markdown(f"- {source}")

def _display_evaluation(eval_scores):
    """Helper to display LLM-as-a-judge evaluation scores."""
    with st.expander("Evaluation Scores"):
        st.markdown(f"**Faithfulness:** {eval_scores.get('faithfulness_score', 'N/A')}/5")
        st.markdown(f"**Relevance:** {eval_scores.get('relevance_score', 'N/A')}/5")
        st.markdown(f"**Reasoning:** {eval_scores.get('reasoning', 'N/A')}")
