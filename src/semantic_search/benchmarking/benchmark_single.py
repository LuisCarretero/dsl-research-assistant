import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
from typing import Optional, Literal
from pathlib import Path
import numpy as np
import json
import re
import os


from semantic_search.data_retrieval.utils import extract_abstract_from_md
from semantic_search.store.faiss_store import FAISSDocumentStore
from semantic_search.store.milvus_store import MilvusDocumentStore
from semantic_search.utils import predict_refs_from_abstract, load_data
from semantic_search.benchmarking.utils import calc_metric_at_topk


def compute_prec_recall_metrics(
    benchmark_data: pd.DataFrame,
    store_name: str,
    store_dirpath: str,
    search_kwargs: dict = {},
    store_type: Literal['faiss', 'milvus'] = 'faiss',
    max_top_k: int = 200
) -> pd.DataFrame:
    """
    benchmark_data should have the following columns:
        text: str
        references: list of str
    """
    # Assert correct benchmark data format
    assert all(col in benchmark_data.columns for col in ['text', 'references'])
    assert isinstance(benchmark_data['references'].iloc[0], list) and isinstance(benchmark_data['references'].iloc[0][0], str)
    assert isinstance(benchmark_data['text'].iloc[0], str)

    # Load document store
    print('Loading document store...')
    if store_type == 'faiss':
        ds = FAISSDocumentStore(db_superdir=store_dirpath, store_name=store_name)
        assert ds.load_store() # Make sure store has been initialized
        NotImplementedError('Implement available_refs for FAISS DS')
    elif store_type == 'milvus':
        ds = MilvusDocumentStore(db_superdir=store_dirpath, store_name=store_name)
        assert ds.load_store() # Make sure store has been initialized
        assert ds.check_index_available()
        available_refs = set([oaid.lower() for oaid in ds.document_store.id.tolist()])

    # Only consider references we have indexed in document store
    benchmark_data['GT_refs'] = benchmark_data['references'].apply(lambda refs: [ref for ref in refs if ref.lower() in available_refs])

    # Predict references
    print('Predicting references...')
    ref_cnts = list(range(1, max_top_k + 1))
    results = []
    try:
        for _, row in tqdm(benchmark_data.iterrows(), total=len(benchmark_data), desc='Predicting references'):
            pred = predict_refs_from_abstract(ds, row['text'], max_top_k=max_top_k, search_kwargs=search_kwargs)
            metrics = calc_metric_at_topk(row['GT_refs'], pred, ref_cnts)
            results.append(metrics)
        results_df = pd.DataFrame(results)
    finally:
        # Close store
        if store_type == 'milvus':
            ds._disconnect_client()

    return results_df


def run_abstract_benchmark(
    experiment_name: str,
    store_name: str,
    metadata_dirpath: str = '/Users/luis/Desktop/ETH/Courses/SS25-DSL/data/metadata3',
    results_dirpath: str = '/Users/luis/Desktop/ETH/Courses/SS25-DSL/results',
    store_dirpath: str = '/Users/luis/Desktop/ETH/Courses/SS25-DSL/db',
    first_n_queries: int = -1,
    store_type: Literal['faiss', 'milvus'] = 'milvus',
    search_kwargs: dict = {},
    max_top_k: int = 200,
):
    # Load paper and reference metadata
    print('Loading data...')
    df, _ = load_data(metadata_dirpath, filter_good_papers=True, filter_good_references=True)
    df['abstract'] = df['fpath'].apply(extract_abstract_from_md)
    df = df[df.abstract.apply(len) > 0]
    df = df.iloc[:first_n_queries]

    df.rename(columns={'abstract': 'text', 'refs_oaids_from_dois': 'references'}, inplace=True)

    results_df = compute_prec_recall_metrics(
        benchmark_data=df,
        store_name=store_name,
        store_dirpath=store_dirpath,
        store_type=store_type,
        search_kwargs=search_kwargs,
        max_top_k=max_top_k,
    )

    Path(results_dirpath).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(f'{results_dirpath}/results_{experiment_name}.csv', index=False)

def run_related_work_benchmark(
        experiment_name: str,
        store_name: str,
        metadata_dirpath: str = '/Users/luis/Desktop/ETH/Courses/SS25-DSL/data/metadata3',
        results_dirpath: str = '/Users/luis/Desktop/ETH/Courses/SS25-DSL/results',
        first_n_queries: int = -1,
        store_dirpath: str = '/Users/luis/Desktop/ETH/Courses/SS25-DSL/db',
        store_type: Literal['faiss', 'milvus'] = 'milvus',
        search_kwargs: dict = {},
        max_top_k: int = 200,
):  
    def remove_intext_citations(text: str) -> str:
    # Pattern matches brackets containing numbers separated by commas and optional spaces
        pattern = r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]'
        return re.sub(pattern, '', text)
    
    with open(os.path.join(metadata_dirpath, 'related_work_data.json'), 'r') as f:
        related_work_data = json.load(f)

    benchmark_data = pd.DataFrame(related_work_data)
    benchmark_data['sentence'] = benchmark_data['sentence'].apply(remove_intext_citations)
    benchmark_data.rename(columns={'sentence': 'text', 'ref_oaids': 'references'}, inplace=True)
    benchmark_data = benchmark_data.iloc[:first_n_queries]

    results_df = compute_prec_recall_metrics(
        benchmark_data=benchmark_data,
        store_name=store_name,
        store_dirpath=store_dirpath,
        store_type=store_type,
        search_kwargs=search_kwargs,
        max_top_k=max_top_k,
    )

    Path(results_dirpath).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(f'{results_dirpath}/results_{experiment_name}.csv', index=False)


def extract_prec_recall_curves(
    results_dirpath: str,
    experiment_name: str,
) -> None:
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
    parser.add_argument('--benchmark_type', type=str, default='abstract', choices=['abstract', 'related_work'])
    parser.add_argument('--do_plotting', type=bool, default=True)

    parser.add_argument('--max_top_k', type=int, default=200)
    parser.add_argument('--first_n_queries', type=int, default=-1)
    parser.add_argument('--metadata_dirpath', type=str, default='/Users/luis/Desktop/ETH/Courses/SS25-DSL/data/metadata3')
    parser.add_argument('--store_dirpath', type=str, default='/Users/luis/Desktop/ETH/Courses/SS25-DSL/db')
    parser.add_argument('--results_dirpath', type=str, default='/Users/luis/Desktop/ETH/Courses/SS25-DSL/benchmark_results')
    parser.add_argument('--store_type', type=str, default='milvus', choices=['faiss', 'milvus'])
    parser.add_argument('--retrieval_method', type=str, default='hybrid', choices=['hybrid', 'embedding', 'keyword'])
    parser.add_argument('--add_hot_papers', type=bool, default=False)
    parser.add_argument('--use_citation_scoring', type=bool, default=True)
    parser.add_argument('--cit_score_weight', type=float, default=0.05)
    parser.add_argument('--hybrid_ranker_type', type=str, default='weighted', choices=['weighted', 'RFR'])
    parser.add_argument('--hybrid_ranker_weights', type=str, default=[0.7, 0.3])
    parser.add_argument('--hybrid_ranker_k', type=int, default=60)
    args = parser.parse_args()

    search_kwargs = dict(
        retrieval_method=args.retrieval_method,
        add_hot_papers=args.add_hot_papers,
        use_citation_scoring=args.use_citation_scoring,
        cit_score_weight=args.cit_score_weight,
        hybrid_ranker={
            'type': args.hybrid_ranker_type, 
            'weights': [float(w) for w in args.hybrid_ranker_weights.strip('[]').split(',')], 
            'k': args.hybrid_ranker_k
        }
    )

    if args.benchmark_type == 'abstract':
        run_abstract_benchmark(
            store_name=args.store_name,
            store_dirpath=args.store_dirpath,
            metadata_dirpath=args.metadata_dirpath,
            results_dirpath=args.results_dirpath,
            experiment_name=args.experiment_name,

            # Settings
            search_kwargs=search_kwargs,
            store_type=args.store_type,
            max_top_k=args.max_top_k,
            first_n_queries=args.first_n_queries
        )
    elif args.benchmark_type == 'related_work':
        run_related_work_benchmark(
            store_name=args.store_name,
            store_dirpath=args.store_dirpath,
            metadata_dirpath=args.metadata_dirpath,
            results_dirpath=args.results_dirpath,
            experiment_name=args.experiment_name,

            # Settings
            first_n_queries=args.first_n_queries,
            store_type=args.store_type,
            search_kwargs=search_kwargs,
            max_top_k=args.max_top_k
        )
    if args.do_plotting:
        extract_prec_recall_curves(
            results_dirpath=args.results_dirpath,
            experiment_name=args.experiment_name
        )

"""
src % python -m semantic_search.benchmarking.benchmark_single --store_name=mini_gte --experiment_name=mini_gte_citReranking --benchmark_type=abstract --first_n_queries=10 --retrieval_method=hybrid
src % python -m semantic_search.benchmarking.benchmark_single --store_name=mini_gte --experiment_name=related_work1 --benchmark_type=related_work --first_n_queries=1000 --max_top_k=30
"""