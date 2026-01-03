import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

def load_model():
    model = SentenceTransformer(MODEL_NAME)
    return model

def embed_texts(model, texts):
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings).tolist()

def cosine_search(query_emb, doc_embs, top_k=3):
    import numpy as np
    q = np.array(query_emb).reshape(1, -1)
    db = np.array(doc_embs)
    sims = cosine_similarity(q, db)[0]
    idxs = list(sims.argsort()[::-1][:top_k])
    scores = [float(sims[i]) for i in idxs]
    return idxs, scores

def save_embeddings(path, docs, embeddings, metadatas):
    payload = {
        "docs": docs,
        "embeddings": embeddings,
        "metadatas": metadatas
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_embeddings(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["docs"], payload["embeddings"], payload.get("metadatas", [])
