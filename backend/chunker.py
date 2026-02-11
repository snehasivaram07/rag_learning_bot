# def chunk_text(text, chunk_size=1000, overlap=200):
#     """
#     Split text into chunks of approximately chunk_size characters,
#     with overlap between chunks to preserve context.
#     """
#     chunks = []
#     start = 0
#     text_length = len(text)

#     while start < text_length:
#         end = min(start + chunk_size, text_length)
#         chunk = text[start:end]
#         chunks.append(chunk.strip())
#         start += chunk_size - overlap  # move start forward with overlap

#     return chunks

# from langchain.text_splitter import RecursiveCharacterTextSplitter


# def chunk_text(pages):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=150,
#         separators=["\n\n", "\n", ".", " "]
#     )

#     chunks = []
#     for p in pages:
#         texts = splitter.split_text(p["text"])
#         for t in texts:
#             chunks.append({
#                 "text": t,
#                 "page": p["page"]
#             })
#     return chunks

# 

def chunk_text(pages, chunk_size=800, overlap=150):
    chunks = []

    for p in pages:
        text = p["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({
                "text": chunk,
                "page": p["page"]
            })
            start += chunk_size - overlap

    return chunks
