import os
import argparse
from typing import Literal

from semantic_search.store.store import FAISSDocumentStore
from semantic_search.store.models import create_embedding_model
from semantic_search.utils import load_metadata


def create_store(
    model_name: str,
    metadata_dirpath: str,
    store_dirpath: str,
    store_name: str | None = None,  # If none, will use model_name.replc
    index_metric: Literal['l2', 'ip'] | None = None,
    max_refs: int = -1,
    store_documents: bool = True,
    store_raw_embeddings: bool = True,
    chunk_store_columns: list[str] = ['doi']
) -> None:
    
    model = create_embedding_model(model_name)
    store = FAISSDocumentStore(
        model, 
        db_dir=os.path.join(store_dirpath, (store_name or model_name.replace("/", "_"))),
        index_metric=index_metric,
        store_documents=store_documents,
        store_raw_embeddings=store_raw_embeddings,
        chunk_store_columns=chunk_store_columns
    )

    if not store.load_store():
        _, ref_df = load_metadata(metadata_dirpath, filter_good_papers=True, filter_good_references=True)
        ref_df.rename(columns={'oaid': 'id', 'abstract': 'text'}, inplace=True)
        store.create_index_from_df(ref_df.iloc[:max_refs])
    else:
        print(f'Store {store_name} already exists in {store_dirpath}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--metadata_dirpath', type=str, default='/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata3')
    parser.add_argument('--store_dirpath', type=str, default='/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/db')
    parser.add_argument('--store_name', type=str, default='main')
    parser.add_argument('--index_metric', type=str, default=None)
    parser.add_argument('--max_refs', type=int, default=-1)
    parser.add_argument('--store_documents', type=bool, default=False)
    parser.add_argument('--store_raw_embeddings', type=bool, default=False)
    parser.add_argument('--chunk_store_columns', type=str, default='doi', 
                       help='Comma-separated list of columns to store in chunk store')
    args = parser.parse_args()

    create_store(
        model_name=args.model_name,
        metadata_dirpath=args.metadata_dirpath,
        store_dirpath=args.store_dirpath,
        store_name=args.store_name,
        index_metric=args.index_metric,
        max_refs=args.max_refs,
        store_documents=args.store_documents,
        store_raw_embeddings=args.store_raw_embeddings,
        chunk_store_columns=args.chunk_store_columns.split(',')
    )
