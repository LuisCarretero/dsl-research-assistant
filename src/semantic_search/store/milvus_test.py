from pymilvus import MilvusClient
from semantic_search.store.models import LocalEmbeddingModel
import numpy as np
import torch

# 1) open (or create) a local Milvus Lite database file
client = MilvusClient("milvus_demo.db")  # embeds etcd+store in-process :contentReference[oaicite:0]{index=0}

# 2) prepare your embedding model and discover dimension
model = LocalEmbeddingModel()
# run a dummy chunk+embed to get embedding dim
_, dummy_encoded = model.chunk_and_encode("hello world")
dummy_emb = model.get_embeddings(dummy_encoded)
EMBED_DIM = dummy_emb.shape[1]

# 3) (re)create your collection
COL_NAME = "docs_chunks"
if client.has_collection(collection_name=COL_NAME):
    client.drop_collection(collection_name=COL_NAME)

client.create_collection(
    collection_name=COL_NAME,
    dimension=EMBED_DIM,   # just specify dim, rest (id→int, vector→float_vector, metric=COSINE) are defaults :contentReference[oaicite:1]{index=1}
)

# 4) helper to ingest raw texts
_next_id = 0
def ingest_texts(texts: list[str]):
    global _next_id
    # chunk & pre-tokenize
    chunks, encoded = model.chunk_and_encode(texts, progress_bar=True)
    
    # Flatten encoded
    tmp = {}
    for single_text_encoded in encoded:
        for k, v in single_text_encoded.items():
            if k not in tmp:
                tmp[k] = []
            tmp[k].append(v)
    encoded_flattened = {k: torch.cat(v) for k, v in tmp.items()}

    embs = model.get_embeddings(encoded_flattened, progress_bar=True)
    # prepare Milvus “entities”
    data = []
    for chunk, emb in zip(chunks, embs):
        data.append({
            "id": _next_id,
            "vector": emb.tolist(),
            "text":  chunk
        })
        _next_id += 1

    res = client.insert(collection_name=COL_NAME, data=data)
    print(f"→ Inserted {res['insert_count']} chunks (IDs {_next_id - len(data)} to {_next_id - 1})")

# 5) simple semantic‐search helper (averaging chunk‐vectors for the query)
def search(query: str, top_k: int = 3):
    # embed query
    _, q_enc = model.chunk_and_encode(query)


    q_embs = model.get_embeddings(q_enc)
    q_vec = q_embs.mean(axis=0).tolist()

    hits = client.search(
        collection_name=COL_NAME,
        data=[q_vec],
        limit=top_k,
        output_fields=["text"]
    )
    for score, entity in [(h["distance"], h["entity"]) for h in hits[0]]:
        print(f"• [{score:.4f}] {entity['text']}")

if __name__ == "__main__":
    docs = [
        "Milvus is an open-source vector database for AI applications.",
        "It can run embedded (Milvus Lite) or as a standalone server."
    ]
    ingest_texts(docs)
    print("\nSearch for “embedded”:")
    search("embedded")
