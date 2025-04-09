import os
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


class LocalEmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        
    def get_embeddings(self, texts):
        # Mean Pooling function
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Process in batches for longer texts
        embeddings = []
        batch_size = 8
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and prepare for the model
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                          max_length=512, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Get model output
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Perform pooling
            batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

class FAISSDocumentStore:
    def __init__(self, embedding_model):
        self.embedding_model: LocalEmbeddingModel = embedding_model
        self.index = None
        self.document_store = None
        self.chunk_store = None  # Initialize chunk_store attribute
        self.index_path = "faiss_document_index"
        self.db_path = "document_store.parquet"
    
    def _chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks"""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) >= chunk_size // 2:  # Only keep chunks of reasonable size
                chunks.append(chunk)
        return chunks
    
    def create_index(self, data_dir):
        """Create FAISS index from documents in the specified directory"""
        documents = []
        document_chunks = []
        chunk_to_doc_id = []
        
        # Process all text files in the directory
        for doc_id, file_path in tqdm(enumerate(list(Path(data_dir).glob("*.txt")))):
            doc_text = file_path.read_text(encoding="utf-8")
            
            documents.append({
                "id": doc_id,
                "name": file_path.name,
                "path": str(file_path),
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
        chunk_store = pd.DataFrame({
            "chunk_id": list(range(len(document_chunks))),
            "doc_id": chunk_to_doc_id,
            "text": document_chunks
        })
        
        # Store the chunk_store in the instance
        self.chunk_store = chunk_store
        
        # Get embeddings for all chunks
        print(f"Generating embeddings for {len(document_chunks)} chunks...")
        embeddings = self.embedding_model.get_embeddings(document_chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIDMap(self.index)
        
        # Add embeddings to the index with their IDs
        self.index.add_with_ids(embeddings, np.array(chunk_store["chunk_id"]).astype('int64'))
        
        # Save the index and document store
        self._save_index_and_store(chunk_store)
        
        print(f"Index created with {len(document_chunks)} chunks from {len(documents)} documents")
        return self.index
    
    def _save_index_and_store(self, chunk_store):
        """Save FAISS index and document store to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else ".", exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        
        # Save document store as parquet
        self.document_store.to_parquet("document_store.parquet")
        
        # Save chunk store as parquet
        chunk_store.to_parquet("chunk_store.parquet")
    
    def load_index(self):
        """Load FAISS index and document store from disk"""
        if os.path.exists(f"{self.index_path}.faiss") and os.path.exists("document_store.parquet") and os.path.exists("chunk_store.parquet"):
            self.index = faiss.read_index(f"{self.index_path}.faiss")
            self.document_store = pd.read_parquet("document_store.parquet")
            self.chunk_store = pd.read_parquet("chunk_store.parquet")
            print(f"Loaded index with {self.index.ntotal} vectors")
            return True
        else:
            print("Index or document store not found")
            return False
    
    def search(self, query, top_k=5):
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
