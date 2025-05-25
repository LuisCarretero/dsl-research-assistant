import os
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Literal, Dict, Any, List, Tuple, Optional
import json
from rank_bm25 import BM25Okapi
import pickle

from src.semantic_search.store.models import LocalEmbeddingModel


class FAISSDocumentStore:
    def __init__(
        self, 
        embedding_model: Optional[LocalEmbeddingModel] = None,
        store_name: str = 'main',
        db_superdir: Optional[str] = None,
        index_metric: Literal['l2', 'ip'] | None = None,
        store_raw_embeddings: bool = False,
        chunk_store_columns: List[str] = [],
        doc_store_columns: List[str] = [],
        use_bm25: bool = True,
    ) -> None:
        """
        Initialize a FAISSDocumentStore.

        Args:
            embedding_model: LocalEmbeddingModel instance
            index_metric: Metric for the FAISS index
            db_dir: Directory to store the database
            store_raw_embeddings: Whether to store raw embeddings
            chunk_store_columns: 
                List of columns to store in the chunk store. This is the data that will be returned by the search method.
                The columns must be present in the document store.
        """

        self.embedding_model = embedding_model
        self.index_metric = index_metric
        if self.embedding_model is not None:
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
        self.db_dir = os.path.join(db_superdir, store_name)
        self.store_name = store_name
        self.store_raw_embeddings = store_raw_embeddings
        self.chunk_store_columns = chunk_store_columns
        self.doc_store_columns = doc_store_columns
        self.use_bm25 = use_bm25

        # Data paths
        self._setup_paths()

    def _setup_paths(self) -> None:
        """Setup paths for the store"""
        if self.db_dir is not None:
            Path(self.db_dir).mkdir(parents=True, exist_ok=True)
            self.metadata_path = os.path.join(self.db_dir, 'metadata.json')
            self.index_path = os.path.join(self.db_dir, 'faiss_document_index.faiss')
            self.doc_store_path = os.path.join(self.db_dir, 'document_store.parquet')
            self.chunk_store_path = os.path.join(self.db_dir, 'chunk_store.parquet')
            self.embeddings_path = os.path.join(self.db_dir, 'embeddings.npy')
            self.bm25_path = os.path.join(self.db_dir, 'bm25.pkl')
        else:
            self.metadata_path = None
            self.index_path = None
            self.doc_store_path = None
            self.chunk_store_path = None
            self.embeddings_path = None
            self.bm25_path = None

    def check_store_exists(self) -> bool:
        """Check if the store exists using metadata file in directory."""
        return os.path.exists(self.metadata_path)
        
    def create_index_from_df(self, documents: pd.DataFrame, db_superdir: Optional[str] = None, store_name: Optional[str] = None, overwrite: bool = False) -> None:
        """
        Create FAISS index from documents preprocessed into a DataFrame. DataFrame must have 
        the following columns: id, text (+ any other metadata columns which will be stored in the document store).
        """
        self._update_name_and_dir(db_superdir, store_name)

        if not overwrite and self.check_store_exists():
            raise ValueError(f"Store {self.store_name} already exists. Set overwrite=True to overwrite.")

        if self.use_bm25:
            self._setup_bm25(documents)

        if 'all' in self.doc_store_columns:
            self.document_store = documents
        else:
            self.document_store = documents[list(set(self.doc_store_columns + ['id']))]

        all_chunks_text, all_chunks_encoded = self.embedding_model.chunk_and_encode(documents['text'].tolist(), progress_bar=True)
        chunk_cnts = [len(chunk) for chunk in all_chunks_text]

        # Get embeddings for all chunks
        print(f"Generating embeddings for {sum(chunk_cnts)} chunks...")
        embeddings = self.embedding_model.get_embeddings(all_chunks_encoded, progress_bar=True)
        if self.store_raw_embeddings:
            self.embeddings = embeddings
        
        # Create chunk store DataFrame
        chunk_store = {
            'chunk_id': list(range(sum(chunk_cnts))),  # FIXME: Change or simplify?
            'doc_id': np.repeat(documents['id'].tolist(), chunk_cnts),
        }

        # Check that all columns in chunk_store_columns are present in the document store
        assert all(col in documents.columns for col in self.chunk_store_columns), \
            f"All columns in chunk_store_columns must be present in the document store"
        
        # Add columns to chunk store
        chunk_store_columns = documents.columns if 'all' in self.chunk_store_columns else self.chunk_store_columns
        for col in chunk_store_columns:
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
        self.save_store()

    def _save_metadata(self) -> None:
        """Save metadata to disk"""
        metadata = {'store': {
            'type': 'faiss',
            'store_name': self.store_name,
            'db_dir': self.db_dir,
            'index_metric': self.index_metric, 
            'store_raw_embeddings': self.store_raw_embeddings,
            'chunk_store_columns': self.chunk_store_columns,
            'doc_store_columns': self.doc_store_columns,
            'use_bm25': self.use_bm25,
        }}
        if self.embedding_model is not None:
            metadata['embedding_model'] = self.embedding_model.get_metadata()
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def _load_metadata(self) -> dict:
        """Load metadata from disk"""
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        store_metadata = metadata.get('store', {})
        assert store_metadata.get('type') == 'faiss', "Metadata file contains invalid store type"
        self.db_dir = store_metadata.get('db_dir')
        self.store_name = store_metadata.get('store_name')
        self.index_metric = store_metadata.get('index_metric')
        self.store_raw_embeddings = store_metadata.get('store_raw_embeddings')
        self.chunk_store_columns = store_metadata.get('chunk_store_columns')
        self.doc_store_columns = store_metadata.get('doc_store_columns')
        self.use_bm25 = store_metadata.get('use_bm25')

        return metadata
    
    def _update_name_and_dir(self, db_superdir: Optional[str] = None, store_name: Optional[str] = None) -> None:
        if db_superdir is not None or store_name is not None:
            self.db_dir = os.path.join(db_superdir, store_name)
            self.store_name = store_name
            self._setup_paths()
        else:
            assert self.db_dir is not None, "db_dir must be provided if not initialized in constructor"
    
    def save_store(self, db_superdir: Optional[str] = None, store_name: Optional[str] = None) -> None:
        """Save FAISS index, document and chunk store to disk"""
        self._update_name_and_dir(db_superdir, store_name)

        # Save metadata
        self._save_metadata()
        
        # Save BM25 model
        if self.use_bm25:
            with open(self.bm25_path, 'wb') as f:
                pickle.dump(self.bm25, f)

        # Save FAISS index and chunk store
        faiss.write_index(self.index, str(self.index_path))
        self.chunk_store.to_parquet(self.chunk_store_path)

        # Optionally, save document store and raw embeddings
        self.document_store.to_parquet(self.doc_store_path)
        if self.store_raw_embeddings:
            np.save(self.embeddings_path, self.embeddings)

    def _verify_db_dir(self) -> None:
        """Verify that the additional data files exist in the database directory"""
        # Required files
        if not os.path.exists(self.doc_store_path):
            raise ValueError(f"Document store path {self.doc_store_path} does not exist.")
        if not os.path.exists(self.index_path):
            raise ValueError(f"Index path {self.index_path} does not exist.")
        if not os.path.exists(self.chunk_store_path):
            raise ValueError(f"Chunk store path {self.chunk_store_path} does not exist.")

        # Optional files
        if self.store_raw_embeddings:
            if not os.path.exists(self.embeddings_path):
                raise ValueError(f"Embeddings path {self.embeddings_path} does not exist.")
        if self.use_bm25:
            if not os.path.exists(self.bm25_path):
                raise ValueError(f"BM25 path {self.bm25_path} does not exist.")

    def load_store(self, db_superdir: Optional[str] = None, store_name: Optional[str] = None, allow_fail: bool = False) -> bool:
        """Load FAISS index and document store from disk"""
        self._update_name_and_dir(db_superdir, store_name)

        # Load metadata
        if not os.path.exists(self.metadata_path):
            if allow_fail:
                return False
            else:
                raise ValueError(f"Metadata file {self.metadata_path} does not exist.")
        metadata = self._load_metadata()

        # Initialize embedding model
        if 'embedding_model' in metadata:
            self.embedding_model = LocalEmbeddingModel(**metadata['embedding_model'])
        else:
            self.embedding_model = None

        # Verify that the database directory exists and contains all required files
        self._verify_db_dir()

        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))
        faiss_metric_type = ['ip', 'l2'][self.index.metric_type]
        if faiss_metric_type != self.index_metric:
            raise ValueError(f"Index metric mismatch between FAISS and store metadata: {faiss_metric_type} != {self.index_metric}")
        
        # Load document and chunk store
        self.document_store = pd.read_parquet(self.doc_store_path)
        self.chunk_store = pd.read_parquet(self.chunk_store_path)

        # Load optional raw embeddings and bm25 object
        if self.store_raw_embeddings:
            self.embeddings = np.load(self.embeddings_path)
        if self.use_bm25:
            with open(self.bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)

        print(f"Loaded index with {self.index.ntotal} vectors")
        return True
    
    def _dist_to_score(self, distance: float | np.ndarray) -> float | np.ndarray:
        """
        Convert distance to similarity score

        Assuming the vectors are normalized, the score is always in [0, 1]
        """
        if self.index_metric == 'l2':
            return 1.0 / (1.0 + distance)
        elif self.index_metric == 'ip':
            return (distance + 1.0) / 2.0
        else:
            raise ValueError(f"Invalid index metric: {self.index_metric}")

    def _merge_chunked_query_results(
        self, 
        distances_matrix: np.ndarray, 
        chunk_ids_matrix: np.ndarray, 
        top_k_docs: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Merges search results from multiple query chunks to find the top unique documents.

        1. Flattening results from all query chunks.
        2. Sorting all retrieved (document_chunk_id, distance) pairs based on distance
           (ascending for L2, descending for IP).
        3. Mapping chunk_ids to their parent doc_ids.
        4. Identifying the best-scoring chunk for each unique document.
        5. Converting these best distances to similarity scores using `self._dist_to_score`.
        6. Returning the top `top_k_docs` documents, sorted by these final scores.

        Args:
            distances_matrix: NumPy array of distances, shape (n_query_chunks, k_retrieved_chunks_per_query_chunk).
                              Distances are from the FAISS index search.
            chunk_ids_matrix: NumPy array of chunk IDs, shape (n_query_chunks, k_retrieved_chunks_per_query_chunk).
                              These are IDs of chunks in the FAISS index.
            top_k_docs: The final number of unique documents to return.

        Output:
            A tuple containing:
            - final_scores: NumPy array of similarity scores for the top documents.
            - final_doc_ids: NumPy array of document IDs for the top documents.
        """



        flat_distances = distances_matrix.flatten()
        flat_chunk_ids = chunk_ids_matrix.flatten()

        # Filter out invalid chunk_ids (e.g., -1 from FAISS if k > ntotal)
        valid_mask = flat_chunk_ids != -1
        if not np.all(valid_mask): # Apply mask only if there are invalid entries
            flat_distances = flat_distances[valid_mask]
            flat_chunk_ids = flat_chunk_ids[valid_mask]

        if len(flat_chunk_ids) == 0:
            return np.array([]), np.array([])

        # Sort based on distances:
        if self.index_metric == 'l2':
            sorted_indices = np.argsort(flat_distances)
        elif self.index_metric == 'ip':
            sorted_indices = np.argsort(flat_distances)[::-1]
        else:
            raise ValueError(f"Invalid index metric: {self.index_metric}")
        sorted_distances = flat_distances[sorted_indices]
        sorted_chunk_ids = flat_chunk_ids[sorted_indices]

        # Get unique chunk IDs that appeared in results to efficiently query chunk_store
        # unique_chunks_in_results will be sorted by chunk_id value.
        # unique_indices_map allows mapping from unique_chunks_in_results back to sorted_chunk_ids.
        # This step is not strictly necessary for correctness if using .reindex below,
        # but can be useful for debugging or alternative mapping strategies.
        # unique_chunks_in_results, _ = np.unique(sorted_chunk_ids, return_inverse=False)

        # Fetch doc_ids for all chunks present in sorted_chunk_ids.
        # self.chunk_store['doc_id'] is a Series indexed by chunk_id.
        # .reindex(sorted_chunk_ids) efficiently maps these doc_ids to the sorted_chunk_ids order.
        # This assumes all chunk_ids in sorted_chunk_ids are valid keys in self.chunk_store.
        doc_ids_for_sorted_chunks = self.chunk_store['doc_id'].reindex(sorted_chunk_ids).values
        
        # Check for NaNs which indicate missing chunk_ids in chunk_store (should not happen in a consistent DB)
        nan_mask = pd.isna(doc_ids_for_sorted_chunks)
        if np.any(nan_mask):
            valid_doc_id_mask = ~nan_mask
            sorted_distances = sorted_distances[valid_doc_id_mask]
            doc_ids_for_sorted_chunks = doc_ids_for_sorted_chunks[valid_doc_id_mask]
            if len(doc_ids_for_sorted_chunks) == 0:
                return np.array([]), np.array([])

        # Find the first occurrence of each doc_id in the score-sorted list.
        _, first_occurrence_indices = np.unique(doc_ids_for_sorted_chunks, return_index=True)
        
        # To maintain the score order, sort these `first_occurrence_indices`.
        final_selection_indices = np.sort(first_occurrence_indices)

        # Get the best distances and corresponding doc_ids for unique documents
        final_best_distances = sorted_distances[final_selection_indices]
        final_doc_ids = doc_ids_for_sorted_chunks[final_selection_indices]

        # Convert these best distances to similarity scores
        final_scores = self._dist_to_score(final_best_distances)
        
        num_results = min(top_k_docs, len(final_scores))
        
        return final_scores[:num_results], final_doc_ids[:num_results]

    def _search_embeddings(self, query: str, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs semantic search using embeddings.

        Returns: Tuple of (scores, document_ids)
        """
        # Get embedding for the query
        _, chunks_encoded = self.embedding_model.chunk_and_encode(query, progress_bar=False)
        query_embedding = self.embedding_model.get_embeddings(chunks_encoded, progress_bar=False)

        # Search in the index
        distances, chunk_ids = self.index.search(query_embedding, top_k)

        # Merge results and convert distances to scores
        return self._merge_chunked_query_results(distances, chunk_ids, top_k)


    def _search_bm25(self, query: str, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs keyword search using BM25.

        Returns: Tuple of (scores, document_ids)
        """
        tokenized_query = query.split()
        if not tokenized_query:
            return np.array([]), np.array([])

        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores
        max_score = np.max(bm25_scores)
        normalized_scores = bm25_scores / max_score if max_score > 0 else bm25_scores

        # Get top k indices
        num_results = min(top_k, len(normalized_scores))
        top_indices = np.argsort(normalized_scores)[::-1][:num_results]

        top_scores = normalized_scores[top_indices]
        top_doc_ids = self.document_store['id'].iloc[top_indices].values

        return top_scores, top_doc_ids

    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        top_m_multiplier: int = 5, 
        rrf_k: int = 60,
        retrieval_method: Literal['embedding', 'keyword', 'hybrid'] = 'hybrid',
        return_scores: bool = False,
        return_doc_metadata: bool = True
    ) -> list[dict]:
        """
        Search for documents similar to the query using a hybrid approach (Embeddings + BM25 if enabled).
        Combines results using Reciprocal Rank Fusion (RRF).

        Args:
            query: The search query string.
            top_k: The final number of top results to return.
            top_m_multiplier: Multiplier for top_k to determine how many results to fetch
                              initially from each search method (embeddings, BM25).
            rrf_k: The ranking constant used in the RRF formula (default is 60).
            retrieval_method: 'embedding', 'keyword', or 'hybrid'.
            return_scores: Whether to include scores in the results.
            return_doc_metadata: Whether to include document metadata from document_store.

        Returns:
            A list of dictionaries, each representing a ranked document with its score
            and metadata.
        """
        if retrieval_method in ['keyword', 'hybrid'] and not self.use_bm25:
            raise ValueError("BM25 is not enabled for this store. Cannot use 'keyword' or 'hybrid' retrieval.")

        if retrieval_method == 'embedding':
            final_scores, final_doc_ids = self._search_embeddings(query, top_k)
        elif retrieval_method == 'keyword':
            final_scores, final_doc_ids = self._search_bm25(query, top_k)
        elif retrieval_method == 'hybrid':
            top_m = top_k * top_m_multiplier

            scores_emb, doc_ids_emb = self._search_embeddings(query, top_m)
            scores_bm25, doc_ids_bm25 = self._search_bm25(query, top_m)

            # Combine results using Reciprocal Rank Fusion (RRF)
            # Create rank maps: {doc_id: rank (0-indexed)}
            rank_emb = {doc_id: i for i, doc_id in enumerate(doc_ids_emb)}
            rank_bm25 = {doc_id: i for i, doc_id in enumerate(doc_ids_bm25)}

            all_doc_ids = set(doc_ids_emb) | set(doc_ids_bm25)

            rrf_scores = {}
            for doc_id in all_doc_ids:
                score = 0.0
                # RRF formula: 1 / (k + rank)
                if doc_id in rank_emb:
                    score += 1.0 / (rrf_k + rank_emb[doc_id])
                if doc_id in rank_bm25:
                    score += 1.0 / (rrf_k + rank_bm25[doc_id])
                rrf_scores[doc_id] = score

            # Sort doc IDs by RRF score in descending order
            sorted_doc_ids = sorted(all_doc_ids, key=lambda doc_id: rrf_scores[doc_id], reverse=True)

            # Select top_k results
            final_doc_ids = sorted_doc_ids[:top_k]
            final_scores = [rrf_scores[doc_id] for doc_id in final_doc_ids]
        else:
            raise ValueError(f"Invalid retrieval_method: {retrieval_method}. Choose from 'embedding', 'keyword', 'hybrid'.")

        # 4. Format results
        results = []
        for i, (score, doc_id) in enumerate(zip(final_scores, final_doc_ids)):
            try:
                # Get chunk information from the store
                doc_row = self.document_store[self.document_store['id'] == doc_id].iloc[0]
                res = {'rank': i + 1, 'id': doc_id}
                if return_scores: res['score'] = float(score)
                if return_doc_metadata: res.update(doc_row.to_dict())
                results.append(res)
            except KeyError:
                print(f"Warning: Chunk ID {doc_id} not found in chunk_store.")
                continue

        return results
    
    def _setup_bm25(self, documents: pd.DataFrame):
        """
        Set up BM25 for keyword search.
        Tokenizes all documents in the chunk store and initializes the BM25 model.
        """
        self.tokenized_corpus = [doc.split() for doc in documents['text'].tolist()]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
