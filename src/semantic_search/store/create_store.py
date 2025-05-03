from semantic_search.store.store import FAISSDocumentStore
from semantic_search.store.models import create_embedding_model
from semantic_search.utils import load_metadata


def create_store() -> None:

    model_name = 'prdev/mini-gte'
    metadata_dir = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata3'

    model = create_embedding_model(model_name)
    store = FAISSDocumentStore(
        model, 
        db_dir=f'/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/db/{model_name.replace("/", "_")}',
        index_metric='ip',
        store_documents=True,
        store_raw_embeddings=True,
        chunk_store_columns=['doi', 'topic']
    )

    if not store.load_store():
        _, ref_df = load_metadata( metadata_dir, filter_good_papers=True, filter_good_references=True)
        ref_df.rename(columns={'oaid': 'id', 'abstract': 'text'}, inplace=True)
        store.create_index_from_df(ref_df)


if __name__ == "__main__":
    create_store()
