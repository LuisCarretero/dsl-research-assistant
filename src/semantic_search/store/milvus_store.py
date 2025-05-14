import pandas as pd
from pathlib import Path
import torch
import numpy as np
from pymilvus import MilvusClient
import os


from semantic_search.store.store import LocalEmbeddingModel


class MilvusDocumentStore:
    def __init__(
        self,
        model: LocalEmbeddingModel,
        db_dir: str,
        collection_name: str = "docs_chunks",
        store_documents: bool = False
    ):
        self.model = model
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.client = None
        self._current_client_db_path = None  # <- Changed only when loading client.
        self.store_documents = store_documents
        self.document_store = None

        # Data paths
        self.doc_store_path: Path | None = None
        self.milvus_db_path: Path | None = None

        self._setup_paths()

    def _setup_paths(self) -> None:
        Path(self.db_dir).mkdir(parents=True, exist_ok=True)
        self.doc_store_path = str(Path(self.db_dir) / 'documents.parquet')
        self.milvus_db_path = str(Path(self.db_dir) / 'milvus.db')
        
    def load_store(self) -> bool:
        """Load existing Milvus collection and document store if enabled."""
        self._load_client()
        
        if self.store_documents:
            if os.path.exists(self.doc_store_path):
                self.document_store = pd.read_parquet(self.doc_store_path)
            
        return self.client.has_collection(collection_name=self.collection_name)

    def _load_client(self) -> None:
        """Load existing Milvus collection and document store if enabled. Doesn't reload if already loaded."""
        if self.milvus_db_path != self._current_client_db_path or self.client is None:
            self.client = MilvusClient(self.milvus_db_path)
            self._current_client_db_path = self.milvus_db_path
        return self.client
        
    def create_index_from_df(self, documents: pd.DataFrame, overwrite: bool = False) -> None:
        """Create new Milvus collection and index documents."""
        
        # Initialize client
        self._setup_paths()
        self._load_client()
        
        # Create collection
        if self.client.has_collection(collection_name=self.collection_name):
            if overwrite:
                self.client.drop_collection(collection_name=self.collection_name)
            else:
                raise ValueError(f"Collection {self.collection_name} already exists. Set overwrite=True to overwrite.")
        
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.model.embedding_dim
        )
        
        # Store documents if enabled
        if self.store_documents:
            documents.to_parquet(self.doc_store_path)
            self.document_store = documents
        
        # Chunk and encode all documents at once
        all_chunks_text, all_chunks_encoded = self.model.chunk_and_encode(documents['text'].tolist(), progress_bar=True)
        
        # Flatten encoded
        tmp = {}
        for single_text_encoded in all_chunks_encoded:
            for k, v in single_text_encoded.items():
                if k not in tmp:
                    tmp[k] = []
                tmp[k].append(v)
        encoded_flattened = {k: torch.cat(v) for k, v in tmp.items()}
        chunks_flattened = [item for sublist in all_chunks_text for item in sublist]
        chunk_cnts = [len(chunk) for chunk in all_chunks_text]
        
        # Get embeddings for all chunks
        print(f"Generating embeddings for {len(list(encoded_flattened.values())[0])} chunks...")
        embeddings = self.model.get_embeddings(encoded_flattened, progress_bar=True)
        
        # Prepare data for insertion
        data = [
            {
                "id": chunk_id,
                "vector": chunk_emb.tolist(),
                "text": chunk_text,
                "doc_id": doc_id
            }
            for chunk_id, (chunk_text, chunk_emb, doc_id) in enumerate(zip(
                chunks_flattened, embeddings, np.repeat(documents['id'].values, chunk_cnts).tolist()
            ))
        ]
        
        # Insert into Milvus
        print(f"Inserting {len(data)} chunks into Milvus...")
        self.client.insert(collection_name=self.collection_name, data=data)
        
        print(f"Indexed {len(documents)} documents in Milvus")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        return_scores: bool = True,
        return_doc_metadata: bool = False
    ) -> list[dict]:
        """Search using dense embeddings."""
        if self.client is None:
            raise ValueError("Store not loaded. Call load_store() first.")
            
        # Encode query
        _, q_enc = self.model.chunk_and_encode(query)
        q_embs = self.model.get_embeddings(q_enc)
        q_vec = q_embs.mean(axis=0).tolist()
        
        # Search in Milvus
        hits = self.client.search(
            collection_name=self.collection_name,
            data=[q_vec],
            limit=top_k,
            output_fields=["text", "doc_id"]
        )
        
        # Format results
        results = []
        for i, (score, entity) in enumerate([(h["distance"], h["entity"]) for h in hits[0]]):
            res = {'rank': i + 1, 'text': entity['text']}
            if return_scores:
                res['score'] = float(score)
                
            # Add document metadata if enabled
            if return_doc_metadata and self.store_documents and self.document_store is not None:
                doc_id = entity['doc_id']
                doc_row = self.document_store[self.document_store['id'] == doc_id].iloc[0]
                res.update(doc_row.to_dict())
                
            results.append(res)
        
        return results
