import os
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
from typing import Literal, Dict, Any, List
import json


from semantic_search.store.models import LocalEmbeddingModel

class FAISSDocumentStore:
    def __init__(
        self, 
        embedding_model: LocalEmbeddingModel | None = None,
        index_metric: Literal['l2', 'ip'] | None = None,
        db_dir: str = '../db',
        store_raw_embeddings: bool = False,
        store_documents: bool = False,
        chunk_store_columns: List[str] = [],
    ) -> None:
        """
        Initialize a FAISSDocumentStore.

        Args:
            embedding_model: LocalEmbeddingModel instance
            index_metric: Metric for the FAISS index
            db_dir: Directory to store the database
            store_raw_embeddings: Whether to store raw embeddings
            store_documents: Whether to store documents
            chunk_store_columns: 
                List of columns to store in the chunk store. This is the data that will be returned by the search method.
                The columns must be present in the document store.
        """

        self.embedding_model = embedding_model
        self.index_metric = index_metric
        if index_metric is None:
            self.index_metric = self.embedding_model.preferred_index_metric
        elif index_metric != self.embedding_model.preferred_index_metric:
            print(f'[WARNING] Specified index metric {index_metric} does not match the embedding model '
                  f'{self.embedding_model.model_name} preferred index metric {self.embedding_model.preferred_index_metric}.')

        # Data: FAISS index, documend and chunk store (both DataFrames)
        self.index = None
        self.document_store = None
        self.chunk_store = None  # Initialize chunk_store attribute

        # Settings
        self.store_raw_embeddings = store_raw_embeddings
        self.store_documents = store_documents
        self.chunk_store_columns = chunk_store_columns

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
        if self.store_documents:
            self.document_store = documents

        all_chunks_text, all_chunks_encoded = self.embedding_model.chunk_and_encode(documents['text'].tolist(), progress_bar=True)
        
        
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
        if self.store_raw_embeddings:
            self.embeddings = embeddings
        
        # Create chunk store DataFrame
        chunk_cnts = [len(chunk) for chunk in all_chunks_text]
        chunk_store = {
            'chunk_id': list(range(sum(chunk_cnts))),  # FIXME: Change or simplify?
            'doc_id': np.repeat(documents['id'].tolist(), chunk_cnts),
        }

        # Check that all columns in chunk_store_columns are present in the document store
        assert all(col in documents.columns for col in self.chunk_store_columns), \
            f"All columns in chunk_store_columns must be present in the document store"
        
        # Add columns to chunk store
        for col in self.chunk_store_columns:
            if col == 'text':  # text is special case as it differs from chunk to chunk for same doc
                chunks_flattened = [item for sublist in all_chunks_text for item in sublist]
                chunk_store['text'] = chunks_flattened
            else:  # FIXME: Might want to force doc store to reduce memory (not repeating this metadata for every chunk)
                chunk_store[col] = np.repeat(documents[col].tolist(), chunk_cnts)
        self.chunk_store = pd.DataFrame(chunk_store).astype({'chunk_id': 'int64'}).set_index('chunk_id')

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
        self.index.add_with_ids(embeddings, np.array(self.chunk_store.index).astype('int64'))
        
        # Save the index and document store
        if write_to_disk:
            self._save_store()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from the store"""
        metadata = {'store': {
            'index_metric': self.index_metric, 
            'store_raw_embeddings': self.store_raw_embeddings,
            'store_documents': self.store_documents,
            'chunk_store_columns': self.chunk_store_columns
        }}
        if self.embedding_model is not None:
            metadata['embedding_model'] = self.embedding_model.get_metadata()
        return metadata
    
    def _save_store(self) -> None:
        """Save FAISS index, document and chunk store to disk"""
        os.makedirs(self.db_dir, exist_ok=False)

        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.get_metadata(), f, indent=4)
        
        # Save FAISS index and chunk store
        faiss.write_index(self.index, str(self.index_path))
        self.chunk_store.to_parquet(self.chunk_store_path)

        # Optionally, save document store and raw embeddings
        if self.store_documents:
            self.document_store.to_parquet(self.document_store_path)
        if self.store_raw_embeddings:
            np.save(self.embeddings_path, self.embeddings)

    def load_store(self) -> bool:
        """Load FAISS index and document store from disk"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            # Settings
            self.index_metric = metadata['store']['index_metric']
            self.store_raw_embeddings = metadata['store']['store_raw_embeddings']
            self.store_documents = metadata['store']['store_documents']
            self.chunk_store_columns = metadata['store']['chunk_store_columns']

            # Initialize embedding model
            if 'embedding_model' in metadata:
                if self.embedding_model is None:
                    self.embedding_model = LocalEmbeddingModel(**metadata['embedding_model'])
                else:
                    print(f'Embedding model already initialized, skipping metadata update')

            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            faiss_metric_type = ['ip', 'l2'][self.index.metric_type]
            if faiss_metric_type != self.index_metric:
                raise ValueError(f"Index metric mismatch between FAISS and store metadata: {faiss_metric_type} != {self.index_metric}")
            
            # Load document and chunk store
            if self.store_documents:
                self.document_store = pd.read_parquet(self.document_store_path)
            self.chunk_store = pd.read_parquet(self.chunk_store_path)

            # Load raw embeddings
            if self.store_raw_embeddings:
                self.embeddings = np.load(self.embeddings_path)

            print(f"Loaded index with {self.index.ntotal} vectors")
            return True
        else:
            print("Index or document store not found")
            return False

    def _merge_query_results(self, scores: np.ndarray, chunk_ids: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Merge results from each chunk into one. TODO: Test other merging strategies"""
        flat_scores = scores.flatten()
        flat_ids = chunk_ids.flatten()
        
        # Create array of indices and sort by decreasing score
        indices = np.argsort(flat_scores)[::-1]
        
        # Get unique chunk_ids while preserving order (first occurrence = best score)
        _, unique_indices = np.unique(flat_ids[indices], return_index=True)
        
        # Sort unique_indices to maintain distance ordering and take top_k
        final_indices = indices[np.sort(unique_indices)][:top_k]
        
        # Get final results
        scores = flat_scores[final_indices]
        chunk_ids = flat_ids[final_indices]

        return scores, chunk_ids
    
    def _dist_to_score(self, distance: float | np.ndarray) -> float | np.ndarray:
        """Convert distance to similarity score"""
        if self.index_metric == 'l2':
            return 1.0 / (1.0 + distance)
        elif self.index_metric == 'ip':
            return distance
        else:
            raise ValueError(f"Invalid index metric: {self.index_metric}")

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
        
        # Convert distances (l2 or ip) to scores (always want to maximise this)
        scores = self._dist_to_score(distances)
        scores, chunk_ids = self._merge_query_results(scores, chunk_ids, top_k)
        
        results = []
        for i, (score, chunk_id) in enumerate(zip(scores, chunk_ids)):
            if chunk_id == -1:  # FAISS returns -1 if there are not enough results
                continue
                
            # Get chunk information
            chunk_row = self.chunk_store.loc[chunk_id]
            res = {"rank": i + 1, "score": score}
            res.update(chunk_row.to_dict())
            results.append(res)
        
        return results
