import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

from semantic_search.data_retrieval.utils import parse_list_string, extract_abstract_from_md
from semantic_search.store.faiss_store import FAISSDocumentStore


def get_good_papers_mask(df: pd.DataFrame) -> np.ndarray:
    # Retrieved title matches query title
    mask = (df.ss_sim_score >= 1)

    # We have OA instances of most references
    mask &= (df.refs_oaids_from_dois.apply(len) / df.ss_ref_cnt) > 0.8

    # If available: SS references agree with OA references (latter often missing)
    mask &= (df.ref_jaccard.fillna(1) > 0.6)
    
    return mask

def get_good_references_mask(df: pd.DataFrame) -> np.ndarray:
    abstract_len = df.abstract.fillna('').apply(len)
    mask = (abstract_len > 0) & (abstract_len < 4000)
    return mask

def load_data(
    dirpath: str,
    filter_good_papers: bool = False,
    filter_good_references: bool = False,
    extract_abstract: bool = False,
    paper_dirpath: str = '/Users/luis/Desktop/ETH/Courses/SS25-DSL/data/Conversions/opencvf-data/txt/'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dirpath = Path(dirpath)
    ref_df = pd.read_csv(dirpath / 'refs.csv')
    df = pd.read_csv(dirpath / 'orig.csv')

    # Parse list strings
    for col in ['oa_refs_oaid', 'ss_refs_doi', 'ss_refs_ssid', 'refs_oaids_from_dois', 'refs_dois_from_oaids']:
        df[col] = df[col].apply(parse_list_string)

    # Filter good papers and references
    if filter_good_papers:
        mask = get_good_papers_mask(df)
        df = df[mask]
    if filter_good_references:
        mask = get_good_references_mask(ref_df)
        ref_df = ref_df[mask]

    # Replace paper directory path
    df['fpath'] = df['fpath'].str.replace(
        '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/Conversions/opencvf-data/txt/', 
        paper_dirpath
    )
    if extract_abstract:
        df['abstract'] = df['fpath'].apply(extract_abstract_from_md)

    return df, ref_df

def predict_refs_from_abstract(
    ds: FAISSDocumentStore, 
    abstract: str, 
    max_top_k: int = 10,
    search_kwargs: dict = {}
) -> List[str]:
    doc_dicts = ds.search(abstract, top_k=max_top_k, return_scores=True, return_doc_metadata=False, **search_kwargs)

    # Docs are sorted by rank by default
    return [doc['id'] for doc in doc_dicts]
