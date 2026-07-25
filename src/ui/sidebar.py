import streamlit as st
import os
from src.ui.styles import LOGO_URL
from src.utils.evaluation import Evaluator

def render_sidebar(rag_pipeline):
    """
    Renders the sidebar for uploading a list of questions to batch evaluate.
    """
    st.sidebar.image(LOGO_URL, width=120)
    st.sidebar.title("Batch Evaluation")
    st.sidebar.markdown("Upload a text file with one question per line to run batch evaluation.")

    uploaded_files = st.sidebar.file_uploader(
        "Choose TXT files", 
        accept_multiple_files=True,
        type=["txt"]
    )

    if st.sidebar.button("Run Batch Evaluation"):
        if uploaded_files:
            # We instantiate the evaluator once
            evaluator = Evaluator()
            
            with st.sidebar.status("Running batch evaluation...", expanded=True) as status:
                for uploaded_file in uploaded_files:
                    content = uploaded_file.getvalue().decode("utf-8")
                    questions = [q.strip() for q in content.split("\n") if q.strip()]
                    
                    st.write(f"Found {len(questions)} questions in {uploaded_file.name}")
                    
                    for idx, q in enumerate(questions):
                        st.write(f"**Q{idx+1}:** {q}")
                        
                        # Query the RAG Pipeline (passing empty history for independent batch questions)
                        result = rag_pipeline.query(q, chat_history=[])
                        answer = result["answer"]
                        
                        # Combine contexts for evaluation
                        context_str = "\n".join([doc.page_content for doc in result["context"]])
                        
                        # Evaluate
                        eval_scores = evaluator.evaluate(question=q, context=context_str, answer=answer)
                        
                        # Display results
                        with st.expander(f"Result for Q{idx+1}", expanded=False):
                            st.markdown(f"**Answer:** {answer}")
                            st.markdown(f"**Faithfulness:** {eval_scores.get('faithfulness_score', 'N/A')}/5")
                            st.markdown(f"**Relevance:** {eval_scores.get('relevance_score', 'N/A')}/5")
                            st.markdown(f"**Reasoning:** {eval_scores.get('reasoning', 'N/A')}")
                            
                status.update(label="Batch evaluation completed!", state="complete", expanded=False)
        else:
            st.sidebar.warning("Please upload a file containing questions first.")
