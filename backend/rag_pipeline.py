

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
