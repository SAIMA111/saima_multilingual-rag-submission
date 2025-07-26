import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from sentence_transformers import SentenceTransformer, util
import pickle
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device set to use", device)

# Load multilingual embedding model (must match with embedder.py)
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Load MT5 QA model
qa_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to(device)
qa_tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
print("✅ Models loaded. Ready to take questions...\n")

# Use absolute paths for safe file access
chunks_path = "/content/multilingual_rag_project/data/chunks.txt"
embeddings_path = "/content/multilingual_rag_project/data/embeddings.pkl"

# Load chunks
if not os.path.exists(chunks_path):
    raise FileNotFoundError(f"❌ chunks.txt not found at {chunks_path}")

with open("multilingual_rag_project/data/chunks.txt", "r", encoding="utf-8") as f:
    chunks = [chunk.strip().replace("\n", " ") for chunk in f.read().split("\n\n") if chunk.strip()]



# Load embeddings
if not os.path.exists(embeddings_path):
    raise FileNotFoundError(f"❌ embeddings.pkl not found at {embeddings_path}")
with open("multilingual_rag_project/data/embeddings.pkl", "rb") as f:
    chunk_embeddings = pickle.load(f)

# Start QA loop
while True:
    query = input("🔍 প্রশ্ন করুন (or type 'exit'): ").strip()
    if query.lower() == "exit":
        break

    # Encode question
    query_embedding = encoder.encode([query], convert_to_tensor=True)

    # Compute similarity
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = torch.topk(scores, k=3)

    print("\n📚 Top Retrieved Chunk:")
    context = chunks[top_k.indices[0]]
    print(context)

    # Prepare mT5 input
    input_text = f"{query} প্রশ্নের উত্তর লেখো। প্রাসঙ্গিক তথ্য: {context}"
    inputs = qa_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Generate answer
    with torch.no_grad():
        output_ids = qa_model.generate(**inputs, max_length=50, num_beams=5)
        answer = qa_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"\n✅ উত্তর: {answer.strip()}\n")
