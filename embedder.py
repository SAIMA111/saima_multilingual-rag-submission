import os
import pickle
from sentence_transformers import SentenceTransformer

# Define absolute paths
chunks_path = "/content/multilingual_rag_project/data/chunks.txt"
embeddings_path = "/content/multilingual_rag_project/data/embeddings.pkl"

# Safety check for file existence
if not os.path.exists(chunks_path):
    raise FileNotFoundError(f"❌ Could not find file: {chunks_path}")

# Load multilingual embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Load and clean the chunks
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]

print(f"📄 {len(chunks)} text chunks loaded. Generating embeddings...")

# Generate embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# Save embeddings
with open(embeddings_path, "wb") as f:
    pickle.dump(embeddings, f)

print(f"✅ {len(embeddings)} embeddings created and saved to:\n{embeddings_path}")
