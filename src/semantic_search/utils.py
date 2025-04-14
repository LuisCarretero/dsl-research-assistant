import pandas as pd
from pathlib import Path
import re
import pyalex
from typing import List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np


pyalex.config.email = "luis.carretero@gmx.de"

def parse_list_string(x: str) -> List[str]:
    """
    Parse a string that contains a list of strings by calling str() on it.
    """
    if pd.isna(x) or x is None:  
        return []
    elif isinstance(x, str):
        # Remove brackets and split by commas, then strip quotes and whitespace
        if x.startswith('[') and x.endswith(']'):
            # Example: "['https://openalex.org/W10789807', 'https://openalex.org/W109508954']"
            items = x[1:-1].split(',')
            return [item.strip().strip("'\"") for item in items if item.strip()]
        return [x]  # If it's a string but not a list format, treat as single item
    return []

def extract_abstract_from_md(fpath: str):
    """
    Extract the abstract from a markdown file created by Docling.
    """
    doc_text = Path(fpath).read_text(encoding="utf-8")
    abstract_match = re.search(r'## Abstract\n\n(.*?)(?=\n\n## \d+\.)', doc_text, re.DOTALL)
    return abstract_match.group(1) if abstract_match else ''

def get_title_from_fpath(fpath: str):
    doc_text = Path(fpath).read_text(encoding="utf-8")
    title_match = re.search(r'## ([^\n#]+)', doc_text)
    return title_match.group(1) if title_match else None

def get_metadata(title: str):
    search_results = pyalex.Works().search(title).select(['id', 'doi', 'referenced_works']).get(page=1, per_page=1)
    return (search_results[0]['doi'], search_results[0]['id'], search_results[0]['referenced_works']) if search_results else (None, None, None)

def multithread_apply(data, func, n_workers: int = 5, progress_bar: bool = True):
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(func, data),
            total=len(data),
            disable=not progress_bar
        ))
    return results

def count_references(row, df):
    total_refs = len(row['referenced_works'])
    dataset_oaids = set(df['oaid'].dropna())
    refs_in_dataset = sum(1 for ref in row['referenced_works'] if ref in dataset_oaids)
    return total_refs, refs_in_dataset

def uninvert_abstract(inv_index):
    l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
    return ' '.join(map(lambda x: x[0], sorted(l_inv, key=lambda x: x[1])))

def get_ref_metadata(ref_works: List[str], progress_bar: bool = False) -> np.ndarray:
    """
    Get metadata of interest for each reference work using OpenAlex API.
    """
    fields_of_interest = ['id', 'abstract_inverted_index', 'type', 'topics']

    if len(ref_works) == 0: return np.array([])
    res = []
    batch_size = 100  # Max PyAlex limit
    batch_cnt = (len(ref_works)-1) // batch_size + 1
    for i in tqdm(range(batch_cnt), disable=not progress_bar):
        batch = ref_works[i*batch_size:(i+1)*batch_size]
        batch = list(map(lambda x: x.split('/')[-1], batch))  # OpenAlex IDs only to reduce request line size (may get bad request if too long)
        raw = pyalex.Works().filter_or(openalex_id=batch).select(fields_of_interest).get(per_page=len(batch))
        for item in raw:
            abstract = uninvert_abstract(item['abstract_inverted_index']) if item['abstract_inverted_index'] is not None else ''
            res.append((
                item['id'], 
                abstract, 
                item['type'], 
                item['topics'][0]['display_name'] if item['topics'] else None,
                item['topics'][0]['domain']['display_name'] if item['topics'] else None,
                item['topics'][0]['field']['display_name'] if item['topics'] else None,
                item['topics'][0]['subfield']['display_name'] if item['topics'] else None
            ))

    return np.array(res)
