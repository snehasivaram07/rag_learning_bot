
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings
