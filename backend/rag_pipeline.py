# import numpy as np
# from backend.embeddings import embed_texts
# from backend.vector_store import load_index

# def answer_question(question: str):
#     index, texts = load_index()

#     q_vec = embed_texts([question])
#     D, I = index.search(np.array(q_vec), k=3)

#     retrieved_chunks = [texts[i] for i in I[0]]
#     context = "\n".join(retrieved_chunks)

#     # Simple grounded answer (no LLM yet)
#     answer = f"Based on the document:\n{context[:500]}..."

#     return answer

# from backend.vector_store import search_index
# from backend.embeddings import embed_texts

# def generate_answer(question, chunks=None):
#     """
#     RAG function to generate an answer from relevant chunks
#     """
#     # If chunks are not provided, retrieve them
#     if chunks is None:
#         chunks = search_index(question, embed_function=embed_texts, persist_directory="vector_store")

#     # Simple answer generation: combine chunks
#     context = " ".join(chunks)
    
#     # Replace this with your LLM call (OpenAI, HuggingFace, etc.)
#     answer = f"Based on the document: {context[:500]}..."  # truncate for demo
    
#     return answer

# from transformers import pipeline
# import numpy as np

# qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# SIMILARITY_THRESHOLD = 0.75

# def answer_question(query, index, chunks, embed_model):
#     q_emb = embed_model.encode([query])
#     distances, indices = index.search(q_emb, k=3)

#     best_score = distances[0][0]
#     retrieved = [chunks[i] for i in indices[0]]

#     context = " ".join([r["text"] for r in retrieved])

#     # Confidence check
#     if best_score < SIMILARITY_THRESHOLD:
#         result = qa(question=query, context=context)
#         return {
#             "answer": result["answer"],
#             "source": retrieved,
#             "mode": "document"
#         }
#     else:
#         # RAG fallback
#         return {
#             "answer":   
#                       qa(question=query, context=context)["answer"],
#             "source": [],
#             "mode": "rag"
#         }

from transformers import pipeline
import numpy as np

qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

SIMILARITY_THRESHOLD = 0.75

def answer_question(query, index, chunks, embed_model):
    q_emb = embed_model.encode([query])
    distances, indices = index.search(q_emb, k=3)

    best_score = distances[0][0]
    retrieved = [chunks[i] for i in indices[0]]

    context = " ".join([r["text"] for r in retrieved])

    # Confidence check
    if best_score < SIMILARITY_THRESHOLD:
        result = qa(question=query, context=context)
        return {
            "answer": result["answer"],
            "source": retrieved,
            "mode": "document"
        }
    else:
        # RAG fallback
        return {
            "answer": "This information is not explicitly present in the document. Based on general knowledge: \n\n" +
                      qa(question=query, context=context)["answer"],
            "source": [],
            "mode": "rag"
        }
