import os
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


class LocalEmbeddingModel:
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 8,
        device: str = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.batch_size = batch_size

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        
    def get_embeddings(self, texts: list[str], progress_bar: bool = False) -> np.ndarray:
        # Process in batches for longer texts
        embeddings = []
        
        if progress_bar:
            iterator = tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings")
        else:
            iterator = range(0, len(texts), self.batch_size)
            
        for i in iterator:
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize and prepare for the model
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                          max_length=512, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Get model output
            with torch.no_grad():
                model_output = self.model(**encoded_input)
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
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200):  # FIXME: Use model chunk size?
        """Split text into overlapping chunks"""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) >= chunk_size // 2:  # Only keep chunks of reasonable size
                chunks.append(chunk)
        return chunks
    
    def create_index(self, data_dir: str) -> faiss.IndexFlatL2:
        """Create FAISS index from documents in the specified directory"""
        documents = []
        document_chunks = []
        chunk_to_doc_id = []
        
        # Process all text files in the directory
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
            
            # Create chunks from the document
            chunks = self._chunk_text(doc_text)
            for chunk in chunks:
                document_chunks.append(chunk)
                chunk_to_doc_id.append(doc_id)
        
        # Create document store DataFrame
        self.document_store = pd.DataFrame(documents)
        
        # Create chunk store DataFrame
        self.chunk_store = pd.DataFrame({
            "chunk_id": list(range(len(document_chunks))),
            "doc_id": chunk_to_doc_id,
            "text": document_chunks
        })
        
        # Get embeddings for all chunks
        print(f"Generating embeddings for {len(document_chunks)} chunks...")
        embeddings = self.embedding_model.get_embeddings(document_chunks, progress_bar=True)
        
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
        query_embedding = self.embedding_model.get_embeddings([query])
        
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
                "document_id": int(doc_id),
                "document_name": doc_row["name"],
                "document_path": doc_row["path"],
                "chunk_text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            })
        
        return results
