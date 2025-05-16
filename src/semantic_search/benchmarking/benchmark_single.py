import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import os
from typing import Optional, Literal
from pathlib import Path
import numpy as np


from semantic_search.data_retrieval.utils import extract_abstract_from_md
from semantic_search.store.store import FAISSDocumentStore
from semantic_search.store.milvus_store import MilvusDocumentStore
from semantic_search.utils import predict_refs_from_abstract, load_metadata
from semantic_search.benchmarking.utils import calc_metric_at_levels
from semantic_search.store.models import LocalEmbeddingModel


def compute_prec_recall_metrics(
    metadata_dirpath: str, 
    store_name: str,
    store_dirpath: str,
    results_dirpath: str,
    experiment_name: Optional[str] = None,
    search_kwargs: dict = {},
    first_n_papers: int = -1,
    store_type: Literal['faiss', 'milvus'] = 'faiss'
) -> None:
    store_name = store_name.replace('/', '_')
    if experiment_name is None:
        print(f'No experiment name provided, using store name: {store_name}')
        experiment_name = store_name

    # Load paper and reference metadata
    print('Loading data...')
    df, ref_df = load_metadata(metadata_dirpath, filter_good_papers=True, filter_good_references=True)
    df['fpath'] = df['fpath'].str.replace('/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant', '/Users/luis/Desktop/ETH/Courses/SS25-DSL')  # FIXME: Do this dynamically?
    df['abstract'] = df['fpath'].apply(extract_abstract_from_md)
    df = df[df.abstract.apply(len) > 0]
    df = df.iloc[:first_n_papers]

    available_refs = set(ref_df['oaid'].str.lower().values)
    df['GT_refs'] = df.refs_oaids_from_dois.apply(lambda refs: [ref for ref in refs if ref in available_refs])
    df['available_ref_ratio'] = df.GT_refs.apply(len) / df.refs_oaids_from_dois.apply(len)

    # Load document store
    print('Loading document store...')
    if store_type == 'faiss':
        ds = FAISSDocumentStore(db_dir=os.path.join(store_dirpath, store_name))
    elif store_type == 'milvus':
        ds = MilvusDocumentStore(
            db_dir=os.path.join(store_dirpath, store_name),
            model=LocalEmbeddingModel()
        )
        assert ds.check_server_health()
    assert ds.load_store() # Make sure store has been initialized

    try:
        # Predict references
        print('Predicting references...')
        ref_cnt = int(df.GT_refs.apply(len).mean())
        max_n_refs = 200
        ref_cnts = list(range(1, max_n_refs + 1))
        
        results = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc='Predicting references'):
            pred = predict_refs_from_abstract(ds, row['abstract'], max_n_refs=max_n_refs, search_kwargs=search_kwargs)
            metrics = calc_metric_at_levels(row['GT_refs'], pred, ref_cnts, ref_cnt)
            results.append(metrics)
        results_df = pd.DataFrame(results)
    finally:
        # Close store
        if store_type == 'milvus':
            ds._disconnect_client()

    # Save results
    Path(results_dirpath).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(f'{results_dirpath}/results_{experiment_name}.csv', index=False)

def extract_prec_recall_curves(
    results_dirpath: str,
    experiment_name: str,
    store_name: str = None
) -> None:
    if experiment_name is None:
        assert store_name is not None, 'No experiment name or store name provided'
        print(f'No experiment name provided, using store name: {store_name}')
        experiment_name = store_name
    df = pd.read_csv(f'{results_dirpath}/results_{experiment_name}.csv')

    # Calculate mean metrics across all samples
    metrics_mean = df.mean(axis=0)
    metrics_err = df.std(axis=0) / np.sqrt(len(df))

    # Extract metrics for different levels
    levels = []
    precision_values, recall_values, f1_values = [], [], []
    precision_err, recall_err, f1_err = [], [], []

    max_level = max([int(name.split('lvl')[-1]) for name in metrics_mean.index.tolist() if name.startswith('prec_lvl')])
    for level in range(1, max_level + 1):  # Assuming levels 1-200 based on output
        level_str = f'lvl{level}'
        if f'prec_{level_str}' in metrics_mean:
            levels.append(level)
            precision_values.append(metrics_mean[f'prec_{level_str}'])
            recall_values.append(metrics_mean[f'rec_{level_str}'])
            f1_values.append(metrics_mean[f'f1_{level_str}'])
            precision_err.append(metrics_err[f'prec_{level_str}'])
            recall_err.append(metrics_err[f'rec_{level_str}'])
            f1_err.append(metrics_err[f'f1_{level_str}'])

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(levels, precision_values, label='Precision', marker='', linewidth=2)
    plt.fill_between(levels, np.array(precision_values) - np.array(precision_err), 
                    np.array(precision_values) + np.array(precision_err), alpha=0.2)
    plt.plot(levels, recall_values, label='Recall', marker='', linewidth=2)
    plt.fill_between(levels, np.array(recall_values) - np.array(recall_err), 
                    np.array(recall_values) + np.array(recall_err), alpha=0.2)
    plt.plot(levels, f1_values, label='F1 Score', marker='', linewidth=2)
    plt.fill_between(levels, np.array(f1_values) - np.array(f1_err), 
                    np.array(f1_values) + np.array(f1_err), alpha=0.2)

    plt.xlabel('Number of References (k)')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score at Different Reference Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(results_dirpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{results_dirpath}/precRecallCurve_{experiment_name}.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--store_name', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=False)
    parser.add_argument('--metadata_dirpath', type=str, default='/Users/luis/Desktop/ETH/Courses/SS25-DSL/raw-data/metadata3')
    parser.add_argument('--store_dirpath', type=str, default='/Users/luis/Desktop/ETH/Courses/SS25-DSL/db')
    parser.add_argument('--results_dirpath', type=str, default='/Users/luis/Desktop/ETH/Courses/SS25-DSL/benchmark_results')
    args = parser.parse_args()

    search_kwargs = {
        # 'retrieval_method': 'hybrid'
        'search_type': 'hybrid'
    }

    # compute_prec_recall_metrics(
    #     metadata_dirpath=args.metadata_dirpath,
    #     store_name=args.store_name,
    #     store_dirpath=args.store_dirpath,
    #     results_dirpath=args.results_dirpath,
    #     experiment_name=args.experiment_name,
    #     search_kwargs=search_kwargs,
    #     store_type='milvus'
    # )
    extract_prec_recall_curves(
        results_dirpath=args.results_dirpath,
        experiment_name=args.experiment_name,
        store_name=args.store_name
    )
