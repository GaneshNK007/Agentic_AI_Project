from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Load docs
with open("data/medical_docs.txt", "r") as f:
    documents = f.readlines()

documents = [doc.strip() for doc in documents if doc.strip()]

doc_embeddings = model.encode(documents)

index = faiss.IndexFlatL2(len(doc_embeddings[0]))
index.add(np.array(doc_embeddings))

def get_answer(query):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k=1)
    return documents[I[0][0]]