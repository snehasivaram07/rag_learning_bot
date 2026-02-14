

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
