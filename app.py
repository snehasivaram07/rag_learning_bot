
import streamlit as st
from transformers import pipeline
from backend.loader import load_pdf_text
from backend.chunker import chunk_text
from backend.embeddings import embed_chunks, model
from backend.vector_store import create_faiss_index
from backend.rag_pipeline import answer_question


@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )


def generate_short_notes(text):
    summarizer = load_summarizer()
    text = text[:3000]  # safety limit
    summary = summarizer(
        text,
        max_length=250,
        min_length=120,
        do_sample=False
    )
    return summary[0]["summary_text"]


st.set_page_config(layout="wide")
st.title(" Study Chatbot ")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "planner" not in st.session_state:
    st.session_state.planner = []

# ---------------- SIDEBAR ----------------
st.sidebar.header(" Upload Document")
pdf = st.sidebar.file_uploader("Upload PDF", type="pdf")

if pdf:
    pages = load_pdf_text(pdf)
    chunks = chunk_text(pages)
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)

    # ✅ STORE IN SESSION STATE (CRITICAL FIX)
    st.session_state["document_text"] = pages
    st.session_state["chunks"] = chunks
    st.session_state["index"] = index

    st.sidebar.success("Document processed")

st.sidebar.markdown("###  Document Tools")

# ---------------- SHORT NOTES BUTTON ----------------
if st.sidebar.button(" Short Notes "):
    if "document_text" not in st.session_state:
        st.sidebar.warning("Please upload a document first.")
    else:
        with st.spinner("Generating short notes..."):
            text = " ".join(page["text"] for page in st.session_state["document_text"])
            notes = generate_short_notes(text)
            st.session_state["short_notes"] = notes

# ---------------- STUDY PLANNER ----------------
st.sidebar.header(" Study Planner ")
topic = st.sidebar.text_input("Topic")
date = st.sidebar.date_input("Date")

if st.sidebar.button("Add"):
    st.session_state.planner.append(f"{date} → {topic}")

for p in st.session_state.planner:
    st.sidebar.write("•", p)

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- SHORT NOTES DISPLAY ----------------
if "short_notes" in st.session_state:
    st.markdown("## Short Notes from Document")
    st.info(st.session_state["short_notes"])

# ---------------- CHAT INPUT (BOTTOM) ----------------
query = st.chat_input("Ask your question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    response = answer_question(
        query,
        st.session_state["index"],
        st.session_state["chunks"],
        model
    )

    answer_text = response["answer"]

    if response["mode"] == "document":
        answer_text += "\n\n📌 **Source:**\n"
        for s in response["source"]:
            answer_text += f"- Page {s['page']}\n"

    with st.chat_message("assistant"):
        st.markdown(answer_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text
    })
