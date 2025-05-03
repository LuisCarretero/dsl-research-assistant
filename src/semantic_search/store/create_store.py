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
    index_metric: Literal['l2', 'ip'] = 'ip',
    max_refs: int = -1
) -> None:
    
    model = create_embedding_model(model_name)
    store = FAISSDocumentStore(
        model, 
        db_dir=os.path.join(store_dirpath, model_name.replace("/", "_")),
        index_metric=index_metric,
        store_documents=True,
        store_raw_embeddings=True,
        chunk_store_columns=['doi', 'topic']
    )

    if not store.load_store():
        _, ref_df = load_metadata(metadata_dirpath, filter_good_papers=True, filter_good_references=True)
        ref_df.rename(columns={'oaid': 'id', 'abstract': 'text'}, inplace=True)
        store.create_index_from_df(ref_df.iloc[:max_refs])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--metadata_dirpath', type=str, default='/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata3')
    parser.add_argument('--store_dirpath', type=str, default='/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/db')
    parser.add_argument('--index_metric', type=str, default='ip')
    parser.add_argument('--max_refs', type=int, default=-1)
    args = parser.parse_args()

    create_store(
        model_name=args.model_name,
        metadata_dirpath=args.metadata_dirpath,
        store_dirpath=args.store_dirpath,
        index_metric=args.index_metric,
        max_refs=args.max_refs
    )
