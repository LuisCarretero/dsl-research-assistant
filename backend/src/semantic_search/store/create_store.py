import os
import argparse
from typing import Literal, List

from src.semantic_search.store.faiss_store import FAISSDocumentStore
from src.semantic_search.store.milvus_store import MilvusDocumentStore
from src.semantic_search.store.models import create_embedding_model
from src.semantic_search.utils import load_data


def create_store(
    model_name: str,
    metadata_dirpath: str,
    store_dirpath: str,
    store_name: str | None = None,  # If none, will use model_name.replc
    index_metric: Literal['l2', 'ip'] | None = None,
    max_refs: int = -1,
    store_raw_embeddings: bool = True,
    doc_store_columns: List[str] = [],
    chunk_store_columns: List[str] = [],
    store_type: Literal['faiss', 'milvus'] = 'faiss',
    overwrite: bool = False,
    store_documents: bool = True,
) -> None:
    store_type = store_type.lower()
    if store_type not in ['faiss', 'milvus']:
        raise ValueError(f"Invalid store type: {store_type}")
    
    model = create_embedding_model(model_name)
    if store_type == 'faiss':
        store = FAISSDocumentStore(
            model, 
            db_superdir=store_dirpath,
            store_name=store_name,
            index_metric=index_metric,
            store_raw_embeddings=store_raw_embeddings,
            chunk_store_columns=chunk_store_columns,
            doc_store_columns=doc_store_columns
        )
    elif store_type == 'milvus':
        store = MilvusDocumentStore(
            model, 
            db_superdir=store_dirpath,
            store_name=store_name,
            store_raw_embeddings=store_raw_embeddings,
            store_documents=store_documents,
        )

    if not store.load_store(allow_fail=True) or overwrite:
        _, ref_df = load_data(metadata_dirpath, filter_good_papers=True, filter_good_references=True)
        ref_df.rename(columns={'oaid': 'id', 'abstract': 'text'}, inplace=True)
        store.create_index_from_df(ref_df.iloc[:max_refs], overwrite=overwrite)
    else:
        print(f'Store {store_name} already exists in {store_dirpath}. Set --overwrite to True to overwrite it.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    basepath = '/Users/luis/Desktop/ETH/Courses/SS25-DSL'
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--metadata_dirpath', type=str, default=os.path.join(basepath, 'data/metadata3'))
    parser.add_argument('--store_dirpath', type=str, default=os.path.join(basepath, 'db'))
    parser.add_argument('--store_name', type=str, default='main')
    parser.add_argument('--index_metric', type=str, default=None)
    parser.add_argument('--max_refs', type=int, default=-1)
    parser.add_argument('--store_raw_embeddings', type=bool, default=False, help='Whether to store raw embeddings in the store. Only relevant for FAISSStore.')
    parser.add_argument('--doc_store_columns', type=str, default=[], nargs='+', 
                       help='List of columns to store in doc store. Only relevant for FAISSStore.')
    parser.add_argument('--chunk_store_columns', type=str, default=[], nargs='+', 
                       help='List of columns to store in chunk store. Only relevant for FAISSStore.')
    parser.add_argument('--store_type', type=str, default='faiss', choices=['faiss', 'milvus'])
    parser.add_argument('--overwrite', action='store_true', default=False, help='Whether to overwrite the store if it already exists')
    parser.add_argument('--store_documents', type=bool, default=True, help='Whether to store documents in the store. Only relevant for MilvusStore.')

    args = parser.parse_args()

    create_store(
        model_name=args.model_name,
        metadata_dirpath=args.metadata_dirpath,
        store_dirpath=args.store_dirpath,
        store_name=args.store_name,
        index_metric=args.index_metric,
        max_refs=args.max_refs,
        store_raw_embeddings=args.store_raw_embeddings,
        doc_store_columns=args.doc_store_columns,
        chunk_store_columns=args.chunk_store_columns,
        store_type=args.store_type,
        overwrite=args.overwrite,
        store_documents=args.store_documents,
    )

# poetry run python -m src.semantic_search.store.create_store --store_type=milvus --store_name=main --store_raw_embeddings=False --overwrite
