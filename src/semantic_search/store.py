import os
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
import sys
from typing import Union, Literal, Dict, Any
import json


class LocalEmbeddingModel:
    def __init__(
        self, 
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        batch_size: int = 8,
        device: str = None,
        pooling_type: Literal['mean', 'last', 'cls'] = 'mean',  # cls is first token
        normalize_embeddings: bool = True
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name

        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        else:
            self.device = torch.device(device)
        print(f'Using device: {self.device}')
        self.model.to(self.device)

        assert pooling_type in ['mean', 'last', 'cls'], f"Invalid pooling type: {pooling_type}"
        self.pooling_type = pooling_type
        self.normalize_embeddings = normalize_embeddings

    def chunk_and_encode(self, texts: Union[list[str], str], progress_bar: bool = False) -> tuple[list[str], list[dict]]:
        """
        Chunk texts at token level using the model's tokenizer and encode the chunks.

        Didnt manage to call tokenizer.prepare_for_model batched so doing weird back and forth
        conversion for now.

        TODO: Allow for single text input (and return single in that case). 
        Allow for np arrays or other iterable inputs.
        """
        if isinstance(texts, str):
            texts = [texts]
            unwrap = True
        else:
            assert isinstance(texts, list), "texts must be a list"
            unwrap = False
        all_chunks_text = []
        all_chunks_encoded = []

        effective_chunk_size = self.chunk_size - 2  # Subtract 2 for the special tokens
        stride = effective_chunk_size - self.chunk_overlap

        # Temporarily increase model_max_length to supress warning message.
        tmp, self.tokenizer.model_max_length = self.tokenizer.model_max_length, sys.maxsize
        texts_tokenized = self.tokenizer(
            texts, 
            add_special_tokens=False, 
            return_token_type_ids=False, 
            return_attention_mask=False,
            padding=False,
        )
        self.tokenizer.model_max_length = tmp

        for text_tokenized in tqdm(texts_tokenized['input_ids'], desc='Chunking and encoding', disable=not progress_bar):
            # Create chunks of tokens
            text_chunks = []
            for i in range(0, len(text_tokenized), stride):
                chunk = text_tokenized[i:i+effective_chunk_size]
                if len(chunk) > 0:  # Only keep non-empty chunks
                    chunk_text = self.tokenizer.decode(chunk)
                    text_chunks.append(chunk_text)

            # Batch encode all chunks for this text
            if text_chunks:
                encoded_chunks = self.tokenizer(
                    text_chunks,
                    padding='max_length',
                    max_length=self.chunk_size,
                    truncation=True,
                    return_tensors='pt'
                )
            else:
                encoded_chunks = None
            
            # Also add to flat lists for backward compatibility
            all_chunks_text.append(text_chunks)
            all_chunks_encoded.append(encoded_chunks)
        
        if unwrap:
            return all_chunks_text[0], all_chunks_encoded[0]
        return all_chunks_text, all_chunks_encoded
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        Same operation as sentence_transformers.model.Pooling(ndim, pooling_mode='mean')
        """
        if attention_mask is None:
            return torch.mean(token_embeddings, dim=1)
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, encoded_inputs: BatchEncoding, progress_bar: bool = False) -> np.ndarray:
        """Generate embeddings from pre-tokenized inputs."""

        embeddings = []
        for i in tqdm(range(0, len(encoded_inputs['input_ids']), self.batch_size), desc="Generating embeddings", disable=not progress_bar):
            batch_encoded = {k: v[i:i+self.batch_size] for k, v in encoded_inputs.items()}

            batch_dict = {k: v.to(self.device) for k, v in batch_encoded.items()}
            
            with torch.no_grad():
                model_output = self.model(**batch_dict)

            if self.pooling_type == 'mean':
                pooled = self._mean_pooling(model_output['last_hidden_state'], batch_dict['attention_mask'])
            elif self.pooling_type == 'last':
                pooled = model_output['last_hidden_state'][:, -1, :]
            elif self.pooling_type == 'cls':
                pooled = model_output['last_hidden_state'][:, 0, :]
            else:
                raise ValueError(f"Invalid pooling type: {self.pooling_type}")
            
            if self.normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            
            embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from the model"""
        return {
            'model_name': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'batch_size': self.batch_size,
            'pooling_type': self.pooling_type,
            'normalize_embeddings': self.normalize_embeddings
        }


class FAISSDocumentStore:
    def __init__(
        self, 
        embedding_model: LocalEmbeddingModel | None = None,
        index_metric: Literal['l2', 'ip'] | None = None,
        db_dir: str = '../db',
    ) -> None:
        self.embedding_model = embedding_model
        self.index_metric = index_metric

        # Data: FAISS index, documend and chunk store (both DataFrames)
        self.index = None
        self.document_store = None
        self.chunk_store = None  # Initialize chunk_store attribute

        # Data paths
        self.db_dir = db_dir
        self.metadata_path = os.path.join(self.db_dir, 'metadata.json')
        self.index_path = os.path.join(self.db_dir, 'faiss_document_index.faiss')
        self.document_store_path = os.path.join(self.db_dir, 'document_store.parquet')
        self.chunk_store_path = os.path.join(self.db_dir, 'chunk_store.parquet')
        self.embeddings_path = os.path.join(self.db_dir, 'embeddings.npy')

    def create_index_from_directory(self, data_dir: str) -> None:
        """Create FAISS index from documents in the specified directory"""
        documents = []

        doc_paths = list(Path(data_dir).glob("*.txt"))
        print(f'Processing {len(doc_paths)} documents...')
        for doc_id, fpath in tqdm(enumerate(doc_paths), total=len(doc_paths), desc="Loading documents"):
            doc_text = fpath.read_text(encoding="utf-8")
            
            documents.append({
                "id": doc_id,
                "name": fpath.name,
                "path": str(fpath),
                "text": doc_text
            })

        self.create_index_from_df(pd.DataFrame(documents))

    def create_index_from_df(self, documents: pd.DataFrame, write_to_disk: bool = True) -> None:
        """
        Create FAISS index from documents preprocessed into a DataFrame. DataFrame must have 
        the following columns: id, text (+ any other metadata columns which will be stored in the document store).
        """

        self.document_store = documents

        all_chunks_text, all_chunks_encoded = self.embedding_model.chunk_and_encode(documents['text'].tolist(), progress_bar=True)
        chunks_flattened = [item for sublist in all_chunks_text for item in sublist]
        
        # Flatten encoded
        tmp = {}
        for single_text_encoded in all_chunks_encoded:
            for k, v in single_text_encoded.items():
                if k not in tmp:
                    tmp[k] = []
                tmp[k].append(v)
        encoded_flattened = {k: torch.cat(v) for k, v in tmp.items()}

        # Get embeddings for all chunks
        print(f"Generating embeddings for {len(list(encoded_flattened.values())[0])} chunks...")
        embeddings = self.embedding_model.get_embeddings(encoded_flattened, progress_bar=True)
        self.embeddings = embeddings
        
        # Create chunk store DataFrame
        self.chunk_store = pd.DataFrame({
            "chunk_id": list(range(len(chunks_flattened))),
            "doc_id": np.repeat(documents['id'].tolist(), [len(chunk) for chunk in all_chunks_text]),
            "text": chunks_flattened
        })

        # Create FAISS index
        emb_dim = embeddings.shape[1]
        if self.index_metric == 'l2':
            self.index = faiss.IndexFlatL2(emb_dim)
        elif self.index_metric == 'ip':
            self.index = faiss.IndexFlatIP(emb_dim)
        else:
            raise ValueError(f"Invalid index metric: {self.index_metric}")
        self.index = faiss.IndexIDMap(self.index)
        
        # Add embeddings to the index with their IDs
        self.index.add_with_ids(embeddings, np.array(self.chunk_store['chunk_id']).astype('int64'))
        
        # Save the index and document store
        if write_to_disk:
            self._save_store()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from the store"""
        metadata = {'store': {'index_metric': self.index_metric}}
        if self.embedding_model is not None:
            metadata['embedding_model'] = self.embedding_model.get_metadata()
        return metadata
    
    def _save_store(self) -> None:
        """Save FAISS index, document and chunk store to disk"""
        os.makedirs(self.db_dir, exist_ok=False)

        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.get_metadata(), f, indent=4)
        
        # Save FAISS index, document and chunk store
        faiss.write_index(self.index, str(self.index_path))
        self.document_store.to_parquet(self.document_store_path)
        self.chunk_store.to_parquet(self.chunk_store_path)
        np.save(self.embeddings_path, self.embeddings)

    def load_store(self) -> bool:
        """Load FAISS index and document store from disk"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            self.index_metric = metadata['store']['index_metric']
            if 'embedding_model' in metadata:
                if self.embedding_model is None:
                    self.embedding_model = LocalEmbeddingModel(**metadata['embedding_model'])
                else:
                    print(f'Embedding model already initialized, skipping metadata update')

            self.index = faiss.read_index(str(self.index_path))
            faiss_metric_type = ['ip', 'l2'][self.index.metric_type]
            if faiss_metric_type != self.index_metric:
                raise ValueError(f"Index metric mismatch between FAISS and store metadata: {faiss_metric_type} != {self.index_metric}")
            self.document_store = pd.read_parquet(self.document_store_path)
            self.chunk_store = pd.read_parquet(self.chunk_store_path)
            self.embeddings = np.load(self.embeddings_path)
            print(f"Loaded index with {self.index.ntotal} vectors")
            return True
        else:
            print("Index or document store not found")
            return False

    def _merge_query_results(self, distances: np.ndarray, chunk_ids: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Merge results from each chunk into one. TODO: Test other merging strategies"""
        flat_dists = distances.flatten()
        flat_ids = chunk_ids.flatten()
        
        # Create array of indices and sort by distance
        indices = np.argsort(flat_dists)
        
        # Get unique chunk_ids while preserving order (first occurrence = best score)
        _, unique_indices = np.unique(flat_ids[indices], return_index=True)
        
        # Sort unique_indices to maintain distance ordering and take top_k
        final_indices = indices[np.sort(unique_indices)][:top_k]
        
        # Get final results
        distances = flat_dists[final_indices]
        chunk_ids = flat_ids[final_indices]

        return distances, chunk_ids

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for documents similar to the query"""
        if self.index is None:
            if not self.load_store():
                raise ValueError("Index not created or loaded")
        
        # Get embedding for the query
        _, chunks_encoded = self.embedding_model.chunk_and_encode(query, progress_bar=False)
        query_embedding = self.embedding_model.get_embeddings(chunks_encoded, progress_bar=False)
        
        # Search in the index
        distances, chunk_ids = self.index.search(query_embedding, top_k)
        distances, chunk_ids = self._merge_query_results(distances, chunk_ids, top_k)
        
        results = []
        for i, (distance, chunk_id) in enumerate(zip(distances, chunk_ids)):
            if chunk_id == -1:  # FAISS returns -1 if there are not enough results
                continue
                
            # Get chunk information
            chunk_row = self.chunk_store[self.chunk_store["chunk_id"] == chunk_id].iloc[0]
            doc_id = chunk_row["doc_id"]
            chunk_text = chunk_row["text"]
            
            results.append({
                "rank": i + 1,
                "score": 1.0 / (1.0 + distance),  # Convert distance to similarity score
                "document_id": doc_id,
                "chunk_text": chunk_text
            })
        
        return results
