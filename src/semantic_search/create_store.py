from semantic_search.store import LocalEmbeddingModel, FAISSDocumentStore
from semantic_search.utils import load_metadata


def create_store() -> None:

    model = LocalEmbeddingModel(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        chunk_size=256,
        chunk_overlap=32,
        batch_size=8
    )
    store = FAISSDocumentStore(
        model, 
        db_dir='/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/db/references-2',
        index_metric='ip'
    )

    if not store.load_index():
        _, ref_df = load_metadata(
            '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata3',
            filter_good_papers=True,
            filter_good_references=True
        )
        ref_df.rename(columns={'oaid': 'id', 'abstract': 'text'}, inplace=True)

        store.create_index_from_df(ref_df)


if __name__ == "__main__":
    create_store()
