
# import os
# import pickle
# import numpy as np
# import faiss

# # -------------------- SAVE VECTOR STORE --------------------
# def save_index(embeddings, texts, persist_directory="vector_store"):
#     """
#     Save embeddings and texts to a FAISS index and pickle file.
#     embeddings: numpy array of shape (num_texts, embedding_dim)
#     texts: list of text chunks
#     """
#     if not os.path.exists(persist_directory):
#         os.makedirs(persist_directory)

#     embeddings = np.array(embeddings).astype('float32')
#     dim = embeddings.shape[1]

#     # Create FAISS index and add embeddings
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)

#     # Save index
#     faiss.write_index(index, os.path.join(persist_directory, "faiss.index"))

#     # Save texts
#     with open(os.path.join(persist_directory, "texts.pkl"), "wb") as f:
#         pickle.dump(texts, f)

# # -------------------- SEARCH VECTOR STORE --------------------
# def search_index(query, embed_function, persist_directory="vector_store", top_k=5):
#     """
#     Search the vector store.
#     query: string query
#     embed_function: function to embed query (e.g., embed_texts)
#     top_k: number of top chunks to return
#     """
#     if embed_function is None:
#         raise ValueError("Please provide an embedding function for the query.")

#     index_path = os.path.join(persist_directory, "faiss.index")
#     texts_path = os.path.join(persist_directory, "texts.pkl")

#     if not os.path.exists(index_path) or not os.path.exists(texts_path):
#         raise FileNotFoundError("Vector store not found. Please process PDFs first.")

#     # Load FAISS index
#     index = faiss.read_index(index_path)
#     with open(texts_path, "rb") as f:
#         texts = pickle.load(f)

#     # Embed query
#     query_embedding = embed_function([query])
#     query_embedding = np.array(query_embedding).astype('float32')

#     # Search FAISS
#     D, I = index.search(query_embedding, top_k)

#     results = [texts[i] for i in I[0]]
#     return results

# 
# import faiss
# import numpy as np

# def create_faiss_index(embeddings):
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)
#     return index

# vector_store.py
import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
