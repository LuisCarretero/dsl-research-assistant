import pandas as pd
from pathlib import Path
import torch
import numpy as np
import os
from typing import Optional, Literal
from collections import defaultdict
from pymilvus import MilvusClient, DataType, Function, FunctionType, WeightedRanker, AnnSearchRequest, RRFRanker


from semantic_search.store.store import LocalEmbeddingModel


class MilvusDocumentStore:
    def __init__(
        self,
        model: LocalEmbeddingModel,
        db_dir: str,
        collection_name: str = "docs_chunks",
        milvus_uri: str = 'http://localhost:19530',
        store_documents: bool = False,
        store_raw_embeddings: bool = False,
    ):
        # Settings
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.store_documents = store_documents
        self.store_raw_embeddings = store_raw_embeddings
        self.milvus_uri = milvus_uri

        # Embedding model
        self.model = model

        # Client and metadata store
        self.client = None
        self.document_store = None

        # Data paths
        self.doc_store_path: Optional[str] = None

        self._setup_paths()

    def _setup_paths(self) -> None:
        Path(self.db_dir).mkdir(parents=True, exist_ok=True)
        self.doc_store_path = str(Path(self.db_dir) / 'documents.parquet')
        self.embeddings_path = str(Path(self.db_dir) / 'embeddings.npy')
        
    def load_store(self) -> bool:
        """Load existing Milvus collection and document store if enabled."""
        self._connect_client()
        
        if self.store_documents:
            if os.path.exists(self.doc_store_path):
                self.document_store = pd.read_parquet(self.doc_store_path)
            else:
                raise ValueError(f"Document store path {self.doc_store_path} does not exist.")
            
        return self.client.has_collection(collection_name=self.collection_name)

    def _connect_client(self) -> None:
        """Connect to Milvus client, with retry logic if the connection fails."""
        if self.client is None:
            try:
                self.client = MilvusClient(self.milvus_uri)
            except Exception as e:
                print(f"Error connecting to Milvus server: {e}")
                print(f"Make sure the Milvus server is running at {self.milvus_uri}")
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

    def _create_collection(self, overwrite: bool = False) -> None:
        """Create new Milvus collection."""
        if self.client.has_collection(collection_name=self.collection_name):
            if overwrite:
                self.client.drop_collection(collection_name=self.collection_name)
            else:
                raise ValueError(f"Collection {self.collection_name} already exists. Set overwrite=True to overwrite.")
        
        # Create schema with both dense and sparse vector fields
        schema = MilvusClient.create_schema()
        
        # Add fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=self.model.embedding_dim)
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
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        
    def create_index_from_df(self, documents: pd.DataFrame, overwrite: bool = False) -> None:
        """Create new Milvus collection and index documents."""
        
        # Initialize client
        self._setup_paths()
        self._connect_client()
        
        # Create collection
        self._create_collection(overwrite=overwrite)
        
        # Store documents if enabled
        if self.store_documents:
            documents.to_parquet(self.doc_store_path)
            self.document_store = documents
        
        # Chunk and encode all documents at once
        all_chunks_text, all_chunks_encoded = self.model.chunk_and_encode(documents['text'].tolist(), progress_bar=True)
        
        # Flatten encoded
        chunks_flattened = [item for sublist in all_chunks_text for item in sublist]
        chunk_cnts = [len(chunk) for chunk in all_chunks_text]
        
        # Get embeddings for all chunks
        print(f"Generating embeddings for {sum(chunk_cnts)} chunks...")
        embeddings = self.model.get_embeddings(all_chunks_encoded, progress_bar=True)

        if self.store_raw_embeddings:
            self.embeddings = embeddings
            np.save(self.embeddings_path, embeddings)
        
        # Prepare data for insertion
        data = [
            {
                "id": chunk_id,
                "dense": chunk_emb.tolist(),
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
    
    def check_server_health(self) -> bool:
        """Check if the Milvus server is healthy and reachable."""
        try:
            if self.client is None:
                self._connect_client()
            # Try a simple operation to check if server is responsive
            _ = self.client.list_collections()
            return True
        except Exception as e:
            print(f"Milvus server health check failed: {e}")
            # Try to reconnect
            self._disconnect_client()
            try:
                self._connect_client()
                return True
            except Exception:
                return False
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        return_scores: bool = True,
        return_doc_metadata: bool = False,
        search_type: Literal['embedding', 'keyword', 'hybrid'] = 'embedding',
        hybrid_ranker: dict = {'type': 'weighted', 'params': {'dense': 0.7, 'sparse': 0.3}}
    ) -> list[dict]:
        """Search using either dense embeddings or keyword search and return document-level results."""
        doc_to_chunk_multiplier = 2
        hybrid_multiplier = 2

        if not self.check_server_health():
            raise ConnectionError("Cannot search - Milvus server is unavailable. Please restart the server.")
        
        if search_type in ['embedding', 'hybrid']:
            # Encode query
            _, q_enc = self.model.chunk_and_encode(query)
            q_embs = self.model.get_embeddings(q_enc)
            q_vec = q_embs.mean(axis=0).tolist()  # FIXME: Handle this differently?

        if search_type == 'embedding':
            hits = self.client.search(
                collection_name=self.collection_name,
                data=[q_vec],
                anns_field='dense',
                limit=top_k * doc_to_chunk_multiplier,
                output_fields=['doc_id']
            )
            
        elif search_type == 'keyword':
            search_params = {
                'params': {'drop_ratio_search': 0.2},
            }
            
            hits = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field='sparse',
                limit=top_k * doc_to_chunk_multiplier,
                search_params=search_params,
                output_fields=['doc_id']
            )

        elif search_type == 'hybrid':
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
                collection_name=self.collection_name,
                reqs=[dense_request, sparse_request],
                ranker=ranker,
                limit=top_k * doc_to_chunk_multiplier,
                output_fields=['doc_id']
            )
        else:
            raise ValueError(f"Invalid search type: {search_type}")
        
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
        
        # Sort documents by score
        doc_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for i, (doc_id, score) in enumerate(doc_rankings[:top_k]):
            res = {'rank': i + 1, 'id': doc_id}
            if return_scores:
                res['score'] = float(score)
                
            # Add document metadata if enabled
            if return_doc_metadata and self.store_documents and self.document_store is not None:
                doc_row = self.document_store[self.document_store['id'] == doc_id].iloc[0]
                res.update(doc_row.to_dict())
                
            results.append(res)
        
        return results

    def __del__(self):
        """Ensure client is disconnected when object is garbage collected."""
        self._disconnect_client()
