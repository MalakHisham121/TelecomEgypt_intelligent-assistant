import streamlit as st

LOGO_URL = "data/we_logo.png"

def apply_theme():
    """Apply the global page configuration and custom CSS styling."""
    st.set_page_config(
        page_title="Telecom Egypt AI Assistant",
        page_icon="📞",
        layout="wide"
    )

    # Minimal CSS tweaks since we use .streamlit/config.toml for main colors
    st.markdown("""
    <style>
        /* Chat messages container styling */
        .stChatMessage {
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            border: 1px solid #eaeaea;
        }
    </style>
    """, unsafe_allow_html=True)
