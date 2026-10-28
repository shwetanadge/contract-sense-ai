import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.pdf_parser import extract_text
from src.ingestion.chunker import chunk_text
from src.retrieval.vector_store import load_bm25
from src.analysis.rag_agent import ask

# Page config
st.set_page_config(
    page_title="Compliance RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Load resources once
@st.cache_resource
def load_resources():
    text = extract_text("data/sample_contracts/gdpr.pdf")
    chunks = chunk_text(text, "GDPR")
    load_bm25(chunks)
    return True

# Initialize
with st.spinner("Loading compliance documents..."):
    load_resources()

# Header
st.title("⚖️ Compliance RAG Assistant")
st.caption("Ask questions about GDPR and HGB regulations. Answers are grounded in official regulation text with citations.")

st.divider()

# Sample question buttons
st.markdown("**Try a sample question:**")
col1, col2, col3 = st.columns(3)

sample_question = None

with col1:
    if st.button("📋 When is a DPO required?"):
        sample_question = "When is a DPO required?"

with col2:
    if st.button("⏰ GDPR breach notification deadline?"):
        sample_question = "GDPR data breach notification deadline?"

with col3:
    if st.button("📚 HGB bookkeeping obligations?"):
        sample_question = "What are the bookkeeping obligations under HGB?"

st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "citations" in message:
            with st.expander("📎 Sources"):
                for citation in message["citations"]:
                    st.markdown(f"- {citation}")

# Handle sample question click
if sample_question:
    st.session_state.messages.append({
        "role": "user",
        "content": sample_question
    })
    with st.chat_message("user"):
        st.markdown(sample_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching regulations..."):
            answer, citations = ask(sample_question)
        st.markdown(answer)
        with st.expander("📎 Sources"):
            for citation in citations:
                st.markdown(f"- {citation}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })

# Chat input
if prompt := st.chat_input("Ask a compliance question..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching regulations..."):
            answer, citations = ask(prompt)
        st.markdown(answer)
        with st.expander("📎 Sources"):
            for citation in citations:
                st.markdown(f"- {citation}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })