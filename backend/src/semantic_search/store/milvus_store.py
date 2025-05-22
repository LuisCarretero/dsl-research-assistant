import pandas as pd
from pathlib import Path
import numpy as np
import os
import json
from typing import Optional, Literal, Dict, Any
from collections import defaultdict
from pymilvus import MilvusClient, DataType, Function, FunctionType, WeightedRanker, AnnSearchRequest, RRFRanker


from semantic_search.store.faiss_store import LocalEmbeddingModel


class MilvusDocumentStore:
    def __init__(
        self,
        embedding_model: Optional[LocalEmbeddingModel] = None,
        db_superdir: Optional[str] = None,
        store_name: str = 'main',
        milvus_uri: str = 'http://localhost:19530',
        store_documents: bool = False,
        store_raw_embeddings: bool = False,
    ):
        # Settings
        if db_superdir is not None:
            self.db_dir = os.path.join(db_superdir, store_name)
        else:
            self.db_dir = None

        assert all(c.isalnum() or (c == '_' and i > 0) for i, c in enumerate(store_name)), "Store name must contain only alphanumeric characters and underscores, no leading underscore"
        self.store_name = store_name
        self.store_documents = store_documents
        self.store_raw_embeddings = store_raw_embeddings
        self.milvus_uri = milvus_uri

        # Embedding model
        self.embedding_model = embedding_model

        # Client and metadata store
        self.client = None
        self.document_store = None
        self.embeddings = None

        # Data paths
        self._setup_paths()

    def _setup_paths(self) -> None:
        if self.db_dir is not None:
            Path(self.db_dir).mkdir(parents=True, exist_ok=True)
            self.metadata_path = str(Path(self.db_dir) / 'metadata.json')
            self.doc_store_path = str(Path(self.db_dir) / 'documents.parquet')
            self.embeddings_path = str(Path(self.db_dir) / 'embeddings.npy')
        else:
            self.metadata_path = None
            self.doc_store_path = None
            self.embeddings_path = None
        
    def _save_metadata(self) -> None:
        """Save metadata to disk"""
        metadata = {'store': {
            'type': 'milvus',
            'db_dir': self.db_dir,
            'store_name': self.store_name,
            'milvus_uri': self.milvus_uri,
            'store_documents': self.store_documents,
            'store_raw_embeddings': self.store_raw_embeddings,
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
        assert store_metadata.get('type') == 'milvus', "Metadata file contains invalid store type"
        self.db_dir = store_metadata.get('db_dir')
        self.store_name = store_metadata.get('store_name')
        self.milvus_uri = store_metadata.get('milvus_uri')
        self.store_documents = store_metadata.get('store_documents')
        self.store_raw_embeddings = store_metadata.get('store_raw_embeddings')
        
        return metadata
    
    def _verify_db_dir(self) -> None:
        """Verify that the additional data files exist in the database directory"""
        # No requires files.
        
        # Optional files
        if self.store_documents:
            if not os.path.exists(self.doc_store_path):
                raise ValueError(f"Document store path {self.doc_store_path} does not exist.")
        if self.store_raw_embeddings:
            if not os.path.exists(self.embeddings_path):
                raise ValueError(f"Embeddings path {self.embeddings_path} does not exist.")
    
    def _update_name_and_dir(self, db_superdir: Optional[str] = None, store_name: Optional[str] = None) -> None:
        if store_name is not None:
            self.store_name = store_name
        if db_superdir is not None:
            self.db_dir = os.path.join(db_superdir, self.store_name)
            self._setup_paths()
        else:
            assert self.db_dir is not None, "db_dir must be provided if not initialized"

    def load_store(self, db_superdir: Optional[str] = None, store_name: Optional[str] = None, allow_fail: bool = False) -> bool:
        """Load existing Milvus collection and document store if enabled."""
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

        # Connect to client and load index
        self._connect_client()
        if not self.check_index_available():
            raise ConnectionError("Cannot load store - Milvus collection is unavailable. " + \
                                  "Please restart the server and check that collection is available.")
        

        # Load additional data
        self._verify_db_dir()
        if self.store_documents:
            self.document_store = pd.read_parquet(self.doc_store_path)
        if self.store_raw_embeddings:
            self.embeddings = np.load(self.embeddings_path)

        print(f"Loaded store from {self.db_dir}")
        return True

    def save_store(self, db_superdir: Optional[str] = None, store_name: Optional[str] = None) -> None:
        """Save store to disk"""
        self._update_name_and_dir(db_superdir, store_name)

        self._save_metadata()
        if self.store_documents:
            self.document_store.to_parquet(self.doc_store_path)
        if self.store_raw_embeddings:
            np.save(self.embeddings_path, self.embeddings)


    def _connect_client(self) -> None:
        """Connect to Milvus client, with retry logic if the connection fails."""
        if self.client is None or not self.check_index_available():
            try:
                self.client = MilvusClient(self.milvus_uri)
            except Exception as e:
                print(f'Error connecting to Milvus server: {e} \n' + \
                      f'Make sure the Milvus server is running at {self.milvus_uri}')
                raise
        return self.client
    
    def _disconnect_client(self) -> None:
        """Disconnect from Milvus client."""
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            finally:
                self.client = None

    def check_index_available(self) -> bool:
        """Check if the Milvus server is healthy and reachable. 
        Try a simple operation to check if server is responsive"""

        if self.client is None: return False

        try:
            if self.store_name in self.client.list_collections():
                return True
            else:
                return False
        except Exception as e:
            print(f"Milvus server health check failed: {e}")
            return False
        
    def _drop_collection(self) -> None:
        """Drop existing Milvus collection."""
        if self.client.has_collection(collection_name=self.store_name):
            self.client.drop_collection(collection_name=self.store_name)
            print(f"Dropped collection {self.store_name}")
        else:
            print(f"Collection {self.store_name} does not exist.")

    def _create_collection(self, overwrite: bool = False) -> None:
        """Create new Milvus collection."""
        if self.client.has_collection(collection_name=self.store_name):
            if overwrite:
                self._drop_collection()
            else:
                raise ValueError(f"Collection {self.store_name} already exists. Set overwrite=True to overwrite.")

        # Create schema with both dense and sparse vector fields
        schema = MilvusClient.create_schema()
        
        # Add fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_model.embedding_dim)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        
        # Add BM25 function for text search
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25
        )
        schema.add_function(bm25_function)

        # Configure index
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name='dense',
            metric_type='COSINE',
            index_type='FLAT',
            # index_type='HNSW',
            # params={
            #     'M': 8,
            #     'efConstruction': 64
            # }
        )
        index_params.add_index(
            field_name='sparse',
            index_type='SPARSE_INVERTED_INDEX',
            metric_type='BM25',
            params={
                'inverted_index_algo': 'DAAT_MAXSCORE',
                'bm25_k1': 1.2,
                'bm25_b': 0.75
            }
        )
        
        # Create collection with schema
        self.client.create_collection(
            collection_name=self.store_name,
            schema=schema,
            index_params=index_params
        )

    def _check_store_exists(self) -> bool:
        """Check if the store exists using metadata file in directory."""
        return os.path.exists(self.metadata_path)
        
    def create_index_from_df(self, documents: pd.DataFrame, db_superdir: Optional[str] = None, store_name: Optional[str] = None, overwrite: bool = False) -> None:
        """Create new Milvus collection and index documents."""
        self._update_name_and_dir(db_superdir, store_name)

        if not overwrite and self._check_store_exists():
            raise ValueError(f"Store {self.store_name} already exists. Set overwrite=True to overwrite.")

        # Initialize client
        self._connect_client()
        
        # Create collection
        self._create_collection(overwrite=overwrite)
        
        # Store documents if enabled
        if self.store_documents:
            documents.to_parquet(self.doc_store_path)
            self.document_store = documents
        
        # Chunk and encode all documents at once
        all_chunks_text, all_chunks_encoded = self.embedding_model.chunk_and_encode(documents['text'].tolist(), progress_bar=True)
        
        # Flatten encoded
        chunks_flattened = [item for sublist in all_chunks_text for item in sublist]
        chunk_cnts = [len(chunk) for chunk in all_chunks_text]
        
        # Get embeddings for all chunks
        print(f"Generating embeddings for {sum(chunk_cnts)} chunks...")
        embeddings = self.embedding_model.get_embeddings(all_chunks_encoded, progress_bar=True)

        if self.store_raw_embeddings:
            self.embeddings = embeddings
            np.save(self.embeddings_path, embeddings)
        
        # Prepare data for insertion
        data = []
        for chunk_id, (chunk_text, chunk_emb, doc_id) in enumerate(zip(
            chunks_flattened, embeddings, np.repeat(documents['id'].values, chunk_cnts).tolist()
        )):
            data.append({
                "id": chunk_id,
                "dense": chunk_emb.tolist(),
                "text": chunk_text,
                "doc_id": doc_id
            })
            
            # Insert in chunks of 4000 records
            if len(data) >= 4000:
                print(f"Inserting batch of {len(data)} chunks into Milvus...")
                self.client.insert(collection_name=self.store_name, data=data)
                data = []
        
        # Insert any remaining data
        if data:
            print(f"Inserting final batch of {len(data)} chunks into Milvus...")
            self.client.insert(collection_name=self.store_name, data=data)

        # Save store
        self.save_store()
            
        print(f"Indexed {len(documents)} documents in Milvus")

    def _get_citation_scores(self, doc_ids: list[str]) -> list[float]:
        if not self.store_documents:
            raise ValueError("Cannot get citation scores - document store is not enabled.")
        assert self.document_store is not None, "Document store is not loaded."
        
        # Load from cache if available
        if not '_citation_scores' in self.document_store.columns:
            # Compute citation scores
            cit_cnt = self.document_store['cited_by_count'].values
            cit_cnt_log = np.log(cit_cnt + 1)
            cit_scores = cit_cnt_log / np.max(cit_cnt_log)

            # Cache citation scores
            self.document_store['_citation_scores'] = cit_scores

        # Get citation scores
        mask = (self.document_store['id'].isin(doc_ids))
        tmp = self.document_store.loc[mask, ['id', '_citation_scores']].set_index('id')
        return [float(tmp.loc[oaid].values[0]) for oaid in doc_ids]

    def _milvus_search(
        self, 
        query: str, 
        top_k: int = 10, 
        retrieval_method: Literal['embedding', 'keyword', 'hybrid'] = 'embedding', 
        hybrid_ranker: dict = {'type': 'weighted', 'weights': [0.7, 0.3]},
        doc_to_chunk_multiplier: int = 2,
        hybrid_multiplier: int = 2,
    ) -> list[dict]:
        """
        Search using dense embeddings, keyword search, or hybrid search and return document-level results.

        Args:
            query: The query string to search for.
            top_k: The number of top results to return.
            retrieval_method: The method to use for retrieval.
            hybrid_ranker: The ranker to use for hybrid retrieval.
            doc_to_chunk_multiplier: The multiplier for the number of chunks to search.
            hybrid_multiplier: The multiplier for the number of chunks to search for hybrid retrieval.
        """

        if not self.check_index_available():
            raise ConnectionError("Cannot search - Milvus collection is unavailable. " + \
                                  "Please restart the server and check that collection is available.")
        
        if retrieval_method in ['embedding', 'hybrid']:
            # Encode query
            _, q_enc = self.embedding_model.chunk_and_encode(query)
            q_embs = self.embedding_model.get_embeddings(q_enc)
            q_vec = q_embs.mean(axis=0).tolist()  # FIXME: Handle this differently?

        if retrieval_method == 'embedding':
            hits = self.client.search(
                collection_name=self.store_name,
                data=[q_vec],
                anns_field='dense',
                limit=top_k * doc_to_chunk_multiplier,
                output_fields=['doc_id']
            )
            
        elif retrieval_method == 'keyword':
            search_params = {
                'params': {'drop_ratio_search': 0.2},
            }
            
            hits = self.client.search(
                collection_name=self.store_name,
                data=[query],
                anns_field='sparse',
                limit=top_k * doc_to_chunk_multiplier,
                search_params=search_params,
                output_fields=['doc_id']
            )

        elif retrieval_method == 'hybrid':
            # Create AnnSearchRequest for dense vectors
            dense_search_params = {
                'data': [q_vec],
                'anns_field': 'dense',
                'param': {
                    'metric_type': 'COSINE',
                    'params': {'nprobe': 10}
                },
                'limit': top_k * hybrid_multiplier * doc_to_chunk_multiplier,
                # 'output_fields': ['doc_id']
            }
            dense_request = AnnSearchRequest(**dense_search_params)
            
            # Create AnnSearchRequest for sparse vectors (keyword search)
            sparse_search_params = {
                'data': [query],
                'anns_field': 'sparse',
                'param': {
                    'metric_type': 'BM25',
                    'params': {'drop_ratio_search': 0.2}
                },
                'limit': top_k * hybrid_multiplier * doc_to_chunk_multiplier,
                # 'output_fields': ['doc_id']
            }
            sparse_request = AnnSearchRequest(**sparse_search_params)
            
            # Define reranker with appropriate weights
            if hybrid_ranker['type'] == 'weighted':
                ranker = WeightedRanker(*hybrid_ranker.get('weights', [0.5, 0.5]))
            elif hybrid_ranker['type'] == 'RFR':
                ranker = RRFRanker(hybrid_ranker.get('k', 60))
            else:
                raise ValueError(f"Invalid hybrid ranker type: {hybrid_ranker['type']}")
            
            # Perform hybrid search combining dense and sparse vectors
            hits = self.client.hybrid_search(
                collection_name=self.store_name,
                reqs=[dense_request, sparse_request],
                ranker=ranker,
                limit=top_k * doc_to_chunk_multiplier,
                output_fields=['doc_id']
            )
        else:
            raise ValueError(f"Invalid search type: {retrieval_method}")
        
        # Aggregate scores by document
        doc_scores = defaultdict(list)
        for hit in hits[0]:
            score = hit['distance']
            doc_id = hit['entity']['doc_id']
            doc_scores[doc_id].append(score)
        
        # Calculate document scores (using max score among chunks)
        doc_rankings = []
        for doc_id, scores in doc_scores.items():
            doc_score = max(scores)  # Use max score among chunks
            doc_rankings.append((doc_id, doc_score))

        return doc_rankings
    
    def _get_hot_paper_ids(self, top_k: int = 100, cache: bool = True) -> list[int]:
        """Get the top-k most cited papers from the document store."""
        if not self.store_documents:
            raise ValueError("Cannot get hot paper IDs - document store is not enabled.")
        assert self.document_store is not None, "Document store is not loaded."

        if cache and hasattr(self, '_hot_ids') and hasattr(self, '_hot_ids_top_k') and self._hot_ids_top_k == top_k:
            return self._hot_ids
        
        hot_ids = self.document_store.nlargest(top_k, 'cited_by_count')['id'].tolist()
        self._hot_ids_top_k = top_k
        if cache:
            self._hot_ids = hot_ids
        
        return hot_ids

    def search(
        self, 
        query: str, 
        top_k: int = 10,
        return_scores: bool = True,
        return_doc_metadata: bool = False,
        retrieval_method: Literal['embedding', 'keyword', 'hybrid'] = 'embedding',
        hybrid_ranker: dict = {'type': 'weighted', 'weights': [0.7, 0.3]},
        use_citation_scoring: bool = True,
        cit_score_weight: float = 0.2,
        add_hot_papers: bool = True,
        doc_to_chunk_multiplier: int = 2,
        hybrid_multiplier: int = 2,
        hot_paper_multiplier: int = 2,
    ) -> list[dict]:
        """Search using either dense embeddings or keyword search and return document-level results."""
        
        # Run Milvus search (on sparse and dense embeddings)
        emb_rankings = self._milvus_search(
            query=query,
            top_k=top_k,
            retrieval_method=retrieval_method,
            hybrid_ranker=hybrid_ranker,
            doc_to_chunk_multiplier=doc_to_chunk_multiplier,
            hybrid_multiplier=hybrid_multiplier,
        )
        
        # 2) Grab your global top-cited papers from your document_store
        hot_ids = self._get_hot_paper_ids(top_k=top_k * hot_paper_multiplier)

        # 3) Build the UNION of IDs
        embed_ids = {doc_id for doc_id, _ in emb_rankings}
        if add_hot_papers:
            candidate_ids = list(embed_ids) + [i for i in hot_ids if i not in embed_ids]
        else:
            candidate_ids = list(embed_ids)

        if use_citation_scoring:
            # 4) For each candidate, fetch its embed score (0 if absent) and its normalized citation score
            cit_scores = self._get_citation_scores(candidate_ids)
            cit_score_map = {i: s for i, s in zip(candidate_ids, cit_scores)}

            # map embed scores
            emb_score_map = dict(emb_rankings)

            # 5) Compute combined score
            reranked = []
            for doc_id in candidate_ids:
                emb_score = emb_score_map.get(doc_id, 0.0)
                cit_score = cit_score_map.get(doc_id, 0.0)
                combined = cit_score_weight * cit_score + (1 - cit_score_weight) * emb_score
                reranked.append((doc_id, combined, (emb_score, cit_score)))
        else:
            reranked = [(doc_id, score, None) for doc_id, score in emb_rankings]

        # Sort by combined score, take top_k, and format output
        reranked.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, (doc_id, scores, partial_scores) in enumerate(reranked[:top_k]):
            res = {'rank': i + 1, 'id': doc_id}
            if return_scores:
                res.update({'score': float(scores)})
                if use_citation_scoring:
                    res.update({'emb_score': float(partial_scores[0]),
                                'cit_score': float(partial_scores[1])})
                
            # Add document metadata if enabled
            if return_doc_metadata and self.store_documents and self.document_store is not None:
                doc_row = self.document_store[self.document_store['id'] == doc_id].iloc[0]
                res.update(doc_row.to_dict())
                
            results.append(res)
        
        return results

    def __del__(self):
        """Ensure client is disconnected when object is garbage collected."""
        if getattr(self, 'client', None) is not None:
            self._disconnect_client()
