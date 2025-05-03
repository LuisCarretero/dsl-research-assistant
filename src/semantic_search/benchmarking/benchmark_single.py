import pandas as pd
from tqdm import tqdm

from semantic_search.data_retrieval.utils import extract_abstract_from_md
from semantic_search.store import FAISSDocumentStore
from semantic_search.create_store import LocalEmbeddingModel
from semantic_search.utils import predict_refs_from_abstract, load_metadata
from semantic_search.benchmarking.utils import score_predictions


def main() -> None:

    # Load paper and reference metadata
    print('Loading data...')
    df, ref_df = load_metadata(
    '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata3',
    filter_good_papers=True,
    filter_good_references=True
    )
    df['abstract'] = df['fpath'].apply(extract_abstract_from_md)
    df = df[df.abstract.apply(len) > 0]

    available_refs = set(ref_df['oaid'].str.lower().values)
    df['GT_refs'] = df.refs_oaids_from_dois.apply(lambda refs: [ref for ref in refs if ref in available_refs])
    df['available_ref_ratio'] = df.GT_refs.apply(len) / df.refs_oaids_from_dois.apply(len)

    # Load document store
    print('Loading document store...')
    model = LocalEmbeddingModel(
        model_name='prdev/mini-gte',  # 'sentence-transformers/all-MiniLM-L6-v2',
        chunk_size=512,
        chunk_overlap=64,
        batch_size=8
    )
    store = FAISSDocumentStore(model, db_dir='/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/db/prdev-mini-gte')
    assert store.load_index() # Make sure store has been initialized

    # Predict references
    print('Predicting references...')
    max_n_refs = int(df.GT_refs.apply(len).mean())
    print(f'Predicting {max_n_refs} references per paper')
    
    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Predicting references'):
        predicted_refs = predict_refs_from_abstract(store, row['abstract'], max_n_refs=max_n_refs)
        results.append(predicted_refs)
    df['predicted_refs'] = results

    # Score predictions
    metrics = df.apply(lambda row: score_predictions(row['GT_refs'], row['predicted_refs']), axis=1)
    metrics_df = pd.DataFrame(metrics.tolist(), columns=['precision', 'recall', 'f1'])
    print('Average metrics:')
    print(metrics_df.mean())


if __name__ == "__main__":
    main()
