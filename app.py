# import streamlit as st
# import os
# import numpy as np

# from backend.loader import load_pdf_text
# from backend.chunker import chunk_text
# from backend.embeddings import embed_texts
# from backend.vector_store import save_index
# from backend.rag_pipeline import answer_question

# st.set_page_config(layout="wide")
# st.title("🤖 RAG Learning Bot")

# # Sidebar
# st.sidebar.header("Upload PDF")
# uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

# if uploaded_file:
#     os.makedirs("data/uploads", exist_ok=True)
#     file_path = f"data/uploads/{uploaded_file.name}"

#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     text = load_pdf_text(file_path)
#     chunks = chunk_text(text)
#     vectors = embed_texts(chunks)

#     save_index(np.array(vectors), chunks)
#     st.sidebar.success("Document indexed successfully!")

# # Chat
# question = st.text_input("Ask a question from the document")

# if question:
#     response = answer_question(question)
#     st.write(response)



# import streamlit as st
# import os
# import numpy as np

# from backend.loader import load_pdf_text
# from backend.chunker import chunk_text
# from backend.embeddings import embed_texts
# from backend.vector_store import save_index,search_index
# from backend.rag_pipeline import generate_answer  # Updated function for RAG

# st.set_page_config(layout="wide")
# st.title("🤖 RAG Learning Bot")

# # -------------------- SIDEBAR: Upload PDF --------------------
# st.sidebar.header("Upload PDF")
# uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

# persist_dir = "vector_store"

# if uploaded_file:
#     os.makedirs("data/uploads", exist_ok=True)
#     file_path = f"data/uploads/{uploaded_file.name}"

#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # -------------------- LOAD AND CHUNK PDF --------------------
#     text = load_pdf_text(file_path)
#     chunks = chunk_text(text, chunk_size=1000, overlap=200)  # Proper chunking
#     embeddings = embed_texts(chunks)

#     # Save vector store for retrieval
#     save_index(np.array(embeddings), chunks, persist_directory=persist_dir)
#     st.sidebar.success("Document indexed successfully!")

# # -------------------- QUESTION INPUT --------------------
# question = st.text_input("Ask a question from the document")

# if question:
#     try:
#         # -------------------- RETRIEVE RELEVANT CHUNKS --------------------
#         relevant_chunks = search_index(
#             query=question,
#             embed_function=embed_texts,  # same embedding function
#             persist_directory=persist_dir,
#             top_k=5
#         )

#         # -------------------- GENERATE ANSWER --------------------
#         answer = generate_answer(question, relevant_chunks)
#         st.write("### Answer:")
#         st.success(answer)

#     except Exception as e:
#         st.error(f"Error: {e}")


# import streamlit as st
# from datetime import datetime, timedelta
# import os
# import numpy as np

# # Backend imports
# from backend.loader import load_pdf_text
# from backend.chunker import chunk_text
# from backend.embeddings import embed_texts
# from backend.vector_store import save_index, search_index
# from backend.rag_pipeline import generate_answer

# st.set_page_config(layout="wide", page_title="RAG Chat Learning Bot")
# st.title("🤖 RAG Chat Learning Bot")

# # ------------------- SIDEBAR: PDF Upload -------------------
# st.sidebar.header("Upload PDF")
# uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

# persist_dir = "vector_store"

# if uploaded_file:
#     os.makedirs("data/uploads", exist_ok=True)
#     file_path = f"data/uploads/{uploaded_file.name}"

#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Chunk and index
#     text = load_pdf_text(file_path)
#     chunks = chunk_text(text, chunk_size=500, overlap=100)
#     embeddings = embed_texts(chunks)
#     save_index(np.array(embeddings), chunks, persist_directory=persist_dir)
#     st.sidebar.success("Document indexed successfully!")

# # ------------------- SESSION STATE -------------------
# if "history" not in st.session_state:
#     st.session_state.history = []

# # ------------------- CHAT INPUT -------------------
# user_input = st.text_input("Type your message here:")

# if st.button("Send") and user_input.strip():
#     # Append user message
#     st.session_state.history.append({"role": "user", "message": user_input})

#     # ------------------- BOT RESPONSE -------------------
#     greetings = ["hi", "hello", "hey"]
#     now = datetime.now().hour

#     if user_input.lower() in greetings:
#         if now < 12:
#             bot_message = "Good morning! How are you today?"
#         elif now < 18:
#             bot_message = "Good afternoon! How are you today?"
#         else:
#             bot_message = "Good evening! How are you today?"
#     else:
#         try:
#             # Retrieve relevant chunks from PDF
#             relevant_chunks = search_index(
#                 query=user_input,
#                 embed_function=embed_texts,
#                 persist_directory=persist_dir,
#                 top_k=5
#             )
#             bot_message = generate_answer(user_input, relevant_chunks)
#         except Exception as e:
#             bot_message = f"Sorry, I could not find an answer. ({e})"

#     st.session_state.history.append({"role": "bot", "message": bot_message})

# # ------------------- DISPLAY MULTI-TURN CHAT -------------------
# st.subheader("💬 Chat")

# for chat in st.session_state.history:
#     if chat["role"] == "user":
#         # User message on the right
#         col1, col2 = st.columns([1, 3])
#         with col1:
#             st.write("")  # empty left
#         with col2:
#             st.markdown(f"<div style='text-align: right; background-color: #DCF8C6; padding:10px; border-radius:10px'>{chat['message']}</div>", unsafe_allow_html=True)
#     else:
#         # Bot message on the left
#         col1, col2 = st.columns([3, 1])
#         with col1:
#             st.markdown(f"<div style='text-align: left; background-color: #F1F0F0; padding:10px; border-radius:10px'>{chat['message']}</div>", unsafe_allow_html=True)
#         with col2:
#             st.write("")  # empty right


# 
# import streamlit as st
# from backend.loader import load_pdf_text
# from backend.chunker import chunk_text
# from backend.embeddings import embed_texts, model
# from backend.vector_store import create_faiss_index
# from backend.rag_pipeline import answer_question

# import streamlit as st
# from backend.loader import load_pdf_text
# from backend.chunker import chunk_text
# from backend.embeddings import embed_texts, model
# from backend.vector_store import create_faiss_index
# from backend.rag_pipeline import answer_question



# st.set_page_config(layout="wide")
# st.title("📘 Intelligent Study Chatbot")

# # ---------------- SESSION STATE ----------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "planner" not in st.session_state:
#     st.session_state.planner = []

# # ---------------- SIDEBAR ----------------
# st.sidebar.header("📄 Upload Document")
# pdf = st.sidebar.file_uploader("Upload PDF", type="pdf")




# if pdf:
#     pages = load_pdf_text(pdf)
#     chunks = chunk_text(pages)
#     embeddings = embed_texts(chunks)
#     index = create_faiss_index(embeddings)
#     st.sidebar.success("Document processed")

# # ---------------- STUDY PLANNER ----------------
# st.sidebar.header("📅 Study Planner")
# topic = st.sidebar.text_input("Topic")
# date = st.sidebar.date_input("Date")

# if st.sidebar.button("Add"):
#     st.session_state.planner.append(f"{date} → {topic}")

# for p in st.session_state.planner:
#     st.sidebar.write("•", p)

# # ---------------- CHAT HISTORY ----------------
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# #

# # ---------------- CHAT INPUT (BOTTOM) ----------------
# query = st.chat_input("Ask your question...")

# if query:
#     st.session_state.messages.append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.markdown(query)

#     response = answer_question(query, index, chunks, model)

#     answer_text = response["answer"]

#     if response["mode"] == "document":
#         answer_text += "\n\n📌 **Source:**\n"
#         for s in response["source"]:
#             answer_text += f"- Page {s['page']}\n"

#     with st.chat_message("assistant"):
#         st.markdown(answer_text)

#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": answer_text
#     })

# import streamlit as st
# from backend.loader import load_pdf_text
# from backend.chunker import chunk_text
# from backend.embeddings import embed_chunks, model
# from backend.vector_store import create_faiss_index
# from backend.rag_pipeline import answer_question

# @st.cache_resource
# def load_summarizer():
#     return pipeline(
#         "summarization",
#         model="facebook/bart-large-cnn"
#     )

# def generate_short_notes(text):
#     summarizer = load_summarizer()
#     text = text[:3000]  # safety limit
#     summary = summarizer(
#         text,
#         max_length=250,
#         min_length=120,
#         do_sample=False
#     )
#     return summary[0]["summary_text"]


# st.set_page_config(layout="wide")
# st.title("📘 Intelligent Study Chatbot")

# # ---------------- SESSION STATE ----------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "planner" not in st.session_state:
#     st.session_state.planner = []

# # ---------------- SIDEBAR ----------------
# st.sidebar.header("📄 Upload Document")
# pdf = st.sidebar.file_uploader("Upload PDF", type="pdf")

# if pdf:
#     pages = load_pdf_text(pdf)
#     chunks = chunk_text(pages)
#     embeddings = embed_chunks(chunks)
#     index = create_faiss_index(embeddings)
#     st.sidebar.success("Document processed")

# st.sidebar.markdown("### 📄 Document Tools")

# # if st.sidebar.button("📝 Short Notes"):
# #     if "document_text" not in st.session_state:
# #         st.sidebar.warning("Please upload a document first.")
#     # else:
#     #     with st.spinner("Generating short notes..."):
#     #         notes = generate_short_notes(
#     #             st.session_state["document_text"]
#     #         )
#     #         st.session_state["short_notes"] = notes
# if st.sidebar.button("📝 Short Notes"):
#     if "chunks" not in st.session_state:
#         st.warning("Please upload a document first.")
#     else:
#         with st.spinner("Generating short notes..."):
#             summarizer = pipeline(
#                 "summarization",
#                 model="facebook/bart-large-cnn"
#             )

#             text = " ".join(st.session_state["document_text"][:3])

#             notes = summarizer(
#                 text,
#                 max_length=200,
#                 min_length=80,
#                 do_sample=False
#             )[0]["summary_text"]

#             st.subheader("📝 Short Notes")
#             st.write(notes)


# # ---------------- STUDY PLANNER ----------------
# st.sidebar.header("📅 Study Planner")
# topic = st.sidebar.text_input("Topic")
# date = st.sidebar.date_input("Date")

# if st.sidebar.button("Add"):
#     st.session_state.planner.append(f"{date} → {topic}")

# for p in st.session_state.planner:
#     st.sidebar.write("•", p)

# # ---------------- CHAT HISTORY ----------------
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
# # ---------------- SHORT NOTES DISPLAY ----------------
# if "short_notes" in st.session_state:
#     st.markdown("## 📝 Short Notes from Document")
#     st.info(st.session_state["short_notes"])

# # ---------------- CHAT INPUT (BOTTOM) ----------------
# query = st.chat_input("Ask your question...")

# if query:
#     st.session_state.messages.append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.markdown(query)

#     response = answer_question(query, index, chunks, model)

#     answer_text = response["answer"]

#     if response["mode"] == "document":
#         answer_text += "\n\n📌 **Source:**\n"
#         for s in response["source"]:
#             answer_text += f"- Page {s['page']}\n"

#     with st.chat_message("assistant"):
#         st.markdown(answer_text)

#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": answer_text
#     })

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
