import os
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from typing import Dict

class LocalEmbeddingModel:
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 256,
        batch_size: int = 8,
        device: str = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.model = AutoModel.from_pretrained(model_name)

        self.batch_size = batch_size
        self.chunk_size = chunk_size

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

    def chunk_texts(self, texts: list[str]) -> tuple[list[str], list[dict]]:
        """
        Chunk texts at token level using the model's tokenizer.
        TODO: Add overlap?
        """
        all_chunks_text = []
        all_chunks_encoded = []

        effective_chunk_size = self.chunk_size - 2  # Subtract 2 for the special tokens

        texts_tokenized = self.tokenizer(texts, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)

        for text_tokenized in texts_tokenized['input_ids']:
            # Create chunks of tokens
            text_chunks, text_encoded_chunks = [], []
            
            for i in range(0, len(text_tokenized), effective_chunk_size):
                chunk = text_tokenized[i:i+effective_chunk_size]
                if len(chunk) > 0:  # Only keep non-empty chunks
                    chunk_text = self.tokenizer.decode(chunk)
                    text_chunks.append(chunk_text)
            
            # Batch encode all chunks for this text
            if text_chunks:
                encoded_chunks = self.tokenizer(
                    text_chunks,
                    padding="max_length",
                    max_length=self.chunk_size,
                    truncation=True,
                    return_tensors="pt"
                )
            
            # Also add to flat lists for backward compatibility
            all_chunks_text.append(text_chunks)
            all_chunks_encoded.append(encoded_chunks)
        
        return all_chunks_text, all_chunks_encoded
    
    def get_embeddings(self, encoded_inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        """Generate embeddings from pre-tokenized inputs."""
        embeddings = []
        
        for i in tqdm(range(0, len(encoded_inputs['input_ids']), self.batch_size), desc="Generating embeddings"):
            batch_encoded = {k: v[i:i+self.batch_size] for k, v in encoded_inputs.items()}

            batch_dict = {k: v.to(self.device) for k, v in batch_encoded.items()}
            
            # Get model output
            with torch.no_grad():
                model_output = self.model(**batch_dict)
            batch_embeddings = model_output.pooler_output
            
            # Normalize embeddings
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)


class FAISSDocumentStore:
    def __init__(
        self, 
        embedding_model: LocalEmbeddingModel,
        db_dir: str = '../db',
    ):
        self.embedding_model = embedding_model

        # Data: FAISS index, documend and chunk store (both DataFrames)
        self.index = None
        self.document_store = None
        self.chunk_store = None  # Initialize chunk_store attribute

        # Data paths
        self.db_dir = db_dir
        self.index_path = os.path.join(self.db_dir, 'faiss_document_index.faiss')
        self.document_store_path = os.path.join(self.db_dir, 'document_store.parquet')
        self.chunk_store_path = os.path.join(self.db_dir, 'chunk_store.parquet')
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200):
        """
        Split text into overlapping chunks
        FIXME: Use model chunk size?
        TODO: Use tokenizer to chunk text
        """
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) >= chunk_size // 2:  # Only keep chunks of reasonable size
                chunks.append(chunk)
        return chunks
    
    def create_index_from_directory(self, data_dir: str):
        """Create FAISS index from documents in the specified directory"""
        documents = []

        doc_paths = list(Path(data_dir).glob("*.txt"))
        print(f'Processing {len(doc_paths)} documents...')
        for doc_id, fpath in tqdm(enumerate(doc_paths), total=len(doc_paths), desc="Chunking documents"):
            doc_text = fpath.read_text(encoding="utf-8")
            
            documents.append({
                "id": doc_id,
                "name": fpath.name,
                "path": str(fpath),
                "text": doc_text
            })

        return pd.DataFrame(documents)

    def create_index(self, documents: pd.DataFrame) -> faiss.IndexFlatL2:
        """
        Create FAISS index from documents preprocessed into a DataFrame.
        DataFrame must have the following columns: id, text (+ any other 
        metadata columns which will be stored in the document store)
        """

        self.document_store = documents

        all_chunks_text, all_chunks_encoded = self.embedding_model.chunk_texts(documents['text'].tolist())
        chunks_flattened = [item for sublist in all_chunks_text for item in sublist]
        
        # Flatten encoded
        tmp = {}
        for single_text_encoded in all_chunks_encoded:
            for k, v in single_text_encoded.items():
                if k not in tmp:
                    tmp[k] = []
                tmp[k].append(v)
        encoded_flattened = {k: torch.cat(v) for k, v in tmp.items()}
        
        # Create chunk store DataFrame
        self.chunk_store = pd.DataFrame({
            "chunk_id": list(range(len(chunks_flattened))),
            "doc_id": np.repeat(documents['id'].tolist(), [len(chunk) for chunk in all_chunks_text]),
            "text": chunks_flattened
        })

        # Get embeddings for all chunks
        print(f"Generating embeddings for {len(encoded_flattened)} chunks...")
        embeddings = self.embedding_model.get_embeddings(encoded_flattened)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIDMap(self.index)
        
        # Add embeddings to the index with their IDs
        self.index.add_with_ids(embeddings, np.array(self.chunk_store["chunk_id"]).astype('int64'))
        
        # Save the index and document store
        self._save_index_and_store()
        
        return self.index
    
    def _save_index_and_store(self) -> None:
        """Save FAISS index, document and chunk store to disk"""
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Save FAISS index, document and chunk store
        faiss.write_index(self.index, self.index_path)
        self.document_store.to_parquet(self.document_store_path)
        self.chunk_store.to_parquet(self.chunk_store_path)
    
    def load_index(self) -> bool:
        """Load FAISS index and document store from disk"""
        if os.path.exists(self.index_path) and os.path.exists(self.document_store_path) and os.path.exists(self.chunk_store_path):
            self.index = faiss.read_index(self.index_path)
            self.document_store = pd.read_parquet(self.document_store_path)
            self.chunk_store = pd.read_parquet(self.chunk_store_path)
            print(f"Loaded index with {self.index.ntotal} vectors")
            return True
        else:
            print("Index or document store not found")
            return False

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for documents similar to the query"""
        if self.index is None:
            if not self.load_index():
                raise ValueError("Index not created or loaded")
        
        # Get embedding for the query
        query_embedding = self.embedding_model.get_embeddings(self.embedding_model.tokenizer(query, return_tensors="pt"))
        
        # Search in the index
        distances, chunk_ids = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (distance, chunk_id) in enumerate(zip(distances[0], chunk_ids[0])):
            if chunk_id == -1:  # FAISS returns -1 if there are not enough results
                continue
                
            # Get chunk information
            chunk_row = self.chunk_store[self.chunk_store["chunk_id"] == chunk_id].iloc[0]
            doc_id = chunk_row["doc_id"]
            chunk_text = chunk_row["text"]
            
            # Get document information
            doc_row = self.document_store[self.document_store["id"] == doc_id].iloc[0]
            
            results.append({
                "rank": i + 1,
                "score": 1.0 / (1.0 + distance),  # Convert distance to similarity score
                "document_id": doc_id,
                "chunk_text": chunk_text
            })
        
        return results
