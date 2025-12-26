import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load embedding model (pre-trained, no training needed)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load text chunks
chunk_texts = []
chunk_names = []

CHUNK_DIR = "rag_main/chunks"

for file in os.listdir(CHUNK_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(CHUNK_DIR, file), "r", encoding="utf-8") as f:
            text = f.read()
            chunk_texts.append(text)
            chunk_names.append(file)

print(f"Loaded {len(chunk_texts)} chunks")

# 3. Convert chunks to embeddings
chunk_embeddings = model.encode(chunk_texts)

# 4. Create FAISS index
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))

print("FAISS index created")

# 5. Search function
def search(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []
    for idx in indices[0]:
        results.append({
            "chunk_name": chunk_names[idx],
            "content": chunk_texts[idx]
        })

    return results


# 6. Test
if __name__ == "__main__":
    query = "seller denied refund"
    results = search(query)

    print("\nTop relevant law sections:\n")
    for r in results:
        print("----", r["chunk_name"], "----")
        print(r["content"][:500])  # preview
        print()
def retrieve_context(query, top_k=3):
    results = search(query, top_k)
    context = "\n\n".join(
        f"Source: {r['chunk_name']}\n{r['content']}"
        for r in results
    )
    return context
