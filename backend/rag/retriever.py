import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==============================================================================
# CLINICAL RAG RETRIEVER MODULE
# - Dense Vector Indexing using SentenceTransformers (all-MiniLM-L6-v2)
# - FAISS IndexFlatIP (Cosine Similarity via Normalized Inner Product)
# - Top-k multi-passage retrieval with similarity thresholding
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "..", "..", "data", "medical_docs.txt")

if not os.path.exists(DOCS_PATH):
    DOCS_PATH = "data/medical_docs.txt"

# Initialize sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load and parse clinical knowledge corpus
documents = []
if os.path.exists(DOCS_PATH):
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            # Skip empty lines and comment headers
            if clean_line and not clean_line.startswith("#"):
                documents.append(clean_line)
else:
    documents = [
        "[WHO-CNS-GLIOMA] Glioma is a primary central nervous system tumor arising from glial cells.",
        "[PCAM-METASTATIC] Metastatic cancer in lymph nodes indicates dissemination of malignant cells requiring oncologic staging.",
        "[HISTO-NORMAL] Normal tissue exhibits intact cellular architecture without neoplastic infiltration."
    ]

# Generate normalized embeddings for cosine similarity
raw_embeddings = model.encode(documents, show_progress_bar=False)
raw_embeddings = np.array(raw_embeddings, dtype=np.float32)

# L2 normalization so Inner Product (IP) equals Cosine Similarity
faiss.normalize_L2(raw_embeddings)
embedding_dim = raw_embeddings.shape[1]

# Build FAISS FlatIP index
index = faiss.IndexFlatIP(embedding_dim)
index.add(raw_embeddings)


def retrieve_context(query: str, top_k: int = 2, min_similarity: float = 0.25):
    """
    Retrieves top_k clinically relevant passages matching the query with cosine similarity scores.
    Returns:
        List of dicts: [{"text": str, "score": float, "id": str}]
    """
    try:
        q_emb = model.encode([query])
        q_emb = np.array(q_emb, dtype=np.float32)
        faiss.normalize_L2(q_emb)

        top_k = min(top_k, len(documents))
        scores, indices = index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= min_similarity:
                doc_text = documents[idx]
                tag = doc_text.split("]")[0].replace("[", "") if "]" in doc_text else "REF"
                results.append({
                    "id": tag,
                    "score": round(float(score), 4),
                    "text": doc_text
                })
        return results
    except Exception as e:
        return [{"id": "ERROR", "score": 0.0, "text": f"Retrieval error: {str(e)}"}]


def get_answer(query: str, top_k: int = 2) -> str:
    """
    Main interface for the Agent and Gemini LLM.
    Returns formatted, multi-chunk clinical literature text.
    """
    results = retrieve_context(query, top_k=top_k)
    if not results or results[0]["id"] == "ERROR":
        return "No specific clinical guideline reference matched the query criteria."

    formatted_context = []
    for r in results:
        formatted_context.append(f"• [{r['id']} | Relevance: {r['score']*100:.1f}%] {r['text']}")

    return "\n".join(formatted_context)


if __name__ == "__main__":
    test_queries = [
        "Glioma WHO grading and neuroimaging features",
        "PCam lymph node metastasis macrometastasis staging",
        "Meningioma dural tail sign and management",
        "Healthy normal lymphoid follicle architecture"
    ]
    print("Testing Clinical RAG Retriever...")
    for q in test_queries:
        print(f"\nQuery: {q}")
        print(get_answer(q, top_k=2))