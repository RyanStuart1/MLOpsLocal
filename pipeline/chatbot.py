import os
import streamlit as st
from pipeline.flow import drift_aware_pipeline
from pipeline.rag_utils import load_and_embed_docs
from pipeline.chat_engine import respond_to_query, build_context, SYSTEM_PROMPT

@st.cache_resource
def get_rag_db():
    return load_and_embed_docs()

# Sidebar Chat UI
def show_chatbot_sidebar():
    st.sidebar.markdown("### Ask the AI Helper")

    model_options = ["gpt-4", "gpt-3.5-turbo"]
    current_model = st.sidebar.selectbox("Choose LLM", model_options, index=0)
    st.sidebar.info(f"Using model: `{current_model}`")

    # File uploader for RAG docs
    st.sidebar.markdown("### 📄 Upload RAG PDF")
    rag_file = st.sidebar.file_uploader("Upload reference PDF", type=["pdf"])

    if rag_file:
        rag_dir = "rag_docs"
        os.makedirs(rag_dir, exist_ok=True)
        rag_path = os.path.join(rag_dir, rag_file.name)
        with open(rag_path, "wb") as f:
            f.write(rag_file.getbuffer())
        st.sidebar.success(f"Uploaded: {rag_file.name}")
        st.cache_resource.clear()
        st.session_state.rag_db = load_and_embed_docs()

    # Ensure RAG DB is loaded
    if "rag_db" not in st.session_state:
        st.session_state.rag_db = get_rag_db()

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Text input and response handling
    user_input = st.sidebar.text_input("Ask a question", key="chat_input")
    send = st.sidebar.button("Send")

    if send and user_input:
        with st.sidebar:
            with st.spinner("Analyzing..."):
                try:
                    context, _, _ = build_context()
                    reply, pdf_path = respond_to_query(
                        user_input=user_input,
                        model=current_model,
                        system_prompt=SYSTEM_PROMPT,
                        context=context,
                        rag_db=st.session_state.rag_db
                    )

                    st.session_state.chat_history.append((user_input, reply))

                    if pdf_path:
                        with open(pdf_path, "rb") as f:
                            st.sidebar.download_button(
                                label="📄 Download Report PDF",
                                data=f,
                                file_name=os.path.basename(pdf_path),
                                mime="application/pdf"
                            )
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

    # Display past exchanges
    if st.session_state.chat_history:
        for user_msg, bot_reply in reversed(st.session_state.chat_history[-5:]):
            st.sidebar.markdown(f"**You:** {user_msg}")
            st.sidebar.markdown(f"**AI:** {bot_reply}")

    # Manual retraining trigger
    if st.sidebar.button("Run Drift Check & Retrain"):
        with st.sidebar:
            with st.spinner("Executing drift-aware pipeline..."):
                result = drift_aware_pipeline()
        if result.get("retrained"):
            st.sidebar.success("Model retrained.")
        else:
            st.sidebar.info("No retraining needed.")