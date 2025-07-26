def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    with open("multilingual_rag_project/data/extracted_text_easyocr.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_text(text)

    with open("multilingual_rag_project/data/chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n{chunk}\n\n")

    print(f"✅ {len(chunks)} chunks created and saved.")
