from semantic_search.store import LocalEmbeddingModel, FAISSDocumentStore
from semantic_search.utils import load_metadata


def create_store() -> None:

    model_name = 'allenai/specter2_base'

    model = LocalEmbeddingModel(
        model_name=model_name,
        chunk_size=512,
        chunk_overlap=64,
        batch_size=8,
        pooling_type='cls',
        normalize_embeddings=True
    )
    store = FAISSDocumentStore(
        model, 
        db_dir=f'/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/db/{model_name.replace("/", "_")}_test1',
        index_metric='ip'
    )

    if not store.load_store():
        _, ref_df = load_metadata(
            '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata3',
            filter_good_papers=True,
            filter_good_references=True
        )
        ref_df.rename(columns={'oaid': 'id', 'abstract': 'text'}, inplace=True)

        store.create_index_from_df(ref_df.iloc[:1000])


if __name__ == "__main__":
    create_store()
