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
from semantic_search.benchmarking.utils import calc_metric_at_topk
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
        ds = FAISSDocumentStore(db_superdir=os.path.join(store_dirpath, store_name))
    elif store_type == 'milvus':
        ds = MilvusDocumentStore(
            db_superdir=os.path.join(store_dirpath, store_name),
            model=LocalEmbeddingModel()
        )
        assert ds._check_index_available()
    assert ds.load_store() # Make sure store has been initialized

    try:
        # Predict references
        print('Predicting references...')
        max_n_refs = 200
        ref_cnts = list(range(1, max_n_refs + 1))
        
        results = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc='Predicting references'):
            pred = predict_refs_from_abstract(ds, row['abstract'], max_n_refs=max_n_refs, search_kwargs=search_kwargs)
            metrics = calc_metric_at_topk(row['GT_refs'], pred, ref_cnts)
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
    ref_cnts = []
    precision_values, recall_values, f1_values, jaccard_values = [], [], [], []
    precision_err, recall_err, f1_err, jaccard_err = [], [], [], []

    max_topk = max([int(name.split('top')[-1]) for name in metrics_mean.index.tolist() if name.startswith('prec_top')])
    for topk in range(1, max_topk + 1):  # Assuming levels 1-200 based on output
        topk_str = f'top{topk}'
        if f'prec_{topk_str}' in metrics_mean:
            ref_cnts.append(topk)
            precision_values.append(metrics_mean[f'prec_{topk_str}'])
            recall_values.append(metrics_mean[f'rec_{topk_str}'])
            f1_values.append(metrics_mean[f'f1_{topk_str}'])
            precision_err.append(metrics_err[f'prec_{topk_str}'])
            recall_err.append(metrics_err[f'rec_{topk_str}'])
            f1_err.append(metrics_err[f'f1_{topk_str}'])
            jaccard_values.append(metrics_mean[f'jaccard_{topk_str}'])
            jaccard_err.append(metrics_err[f'jaccard_{topk_str}'])

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(ref_cnts, precision_values, label='Precision', marker='', linewidth=2)
    plt.fill_between(ref_cnts, np.array(precision_values) - np.array(precision_err), 
                    np.array(precision_values) + np.array(precision_err), alpha=0.2)
    plt.plot(ref_cnts, recall_values, label='Recall', marker='', linewidth=2)
    plt.fill_between(ref_cnts, np.array(recall_values) - np.array(recall_err), 
                    np.array(recall_values) + np.array(recall_err), alpha=0.2)
    plt.plot(ref_cnts, f1_values, label='F1 Score', marker='', linewidth=2)
    plt.fill_between(ref_cnts, np.array(f1_values) - np.array(f1_err), 
                    np.array(f1_values) + np.array(f1_err), alpha=0.2)
    plt.plot(ref_cnts, jaccard_values, label='Jaccard', marker='', linewidth=2)
    plt.fill_between(ref_cnts, np.array(jaccard_values) - np.array(jaccard_err), 
                    np.array(jaccard_values) + np.array(jaccard_err), alpha=0.2)
    
    # Find the index of maximum F1 score
    max_f1_idx = np.argmax(f1_values)
    max_f1_topk = ref_cnts[max_f1_idx]
    max_f1 = f1_values[max_f1_idx]
    max_precision = precision_values[max_f1_idx]
    max_recall = recall_values[max_f1_idx]
    max_jaccard = jaccard_values[max_f1_idx]
    
    # Add vertical line at maximum F1 score
    plt.axvline(x=max_f1_topk, color='gray', linestyle='--', alpha=0.7)
    
    # Add text box with metrics at maximum F1
    textstr = f'Max F1 at top-{max_f1_topk}:\nF1: {max_f1:.3f}\nPrecision: {max_precision:.3f}\nRecall: {max_recall:.3f}\nJaccard: {max_jaccard:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='bottom', bbox=props)

    plt.xlabel('Number of References (k)')
    plt.ylabel('Score')
    plt.title('Precision, Recall, F1 Score, and Jaccard for different top-k references')
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

    compute_prec_recall_metrics(
        metadata_dirpath=args.metadata_dirpath,
        store_name=args.store_name,
        store_dirpath=args.store_dirpath,
        results_dirpath=args.results_dirpath,
        experiment_name=args.experiment_name,
        search_kwargs=search_kwargs,
        store_type='milvus'
    )
    extract_prec_recall_curves(
        results_dirpath=args.results_dirpath,
        experiment_name=args.experiment_name,
        store_name=args.store_name
    )
