import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from semantic_search.data_retrieval.utils import parse_list_string


def get_good_papers_mask(df: pd.DataFrame) -> np.ndarray:
    # Retrieved title matches query title
    mask = (df.ss_sim_score >= 1)

    # We have OA instances of most references
    mask &= (df.refs_oaids_from_dois.apply(len) / df.ss_ref_cnt) > 0.8

    # If available: SS references agree with OA references (latter often missing)
    mask &= (df.ref_jaccard.fillna(1) > 0.6)
    
    return mask

def get_good_references_mask(df: pd.DataFrame) -> np.ndarray:
    mask = (df.abstract.fillna('').apply(len) > 0)
    return mask

def load_metadata(
        dirpath: str,
        filter_good_papers: bool = False,
        filter_good_references: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dirpath = Path(dirpath)
    ref_df = pd.read_csv(dirpath / 'refs.csv')
    df = pd.read_csv(dirpath / 'orig.csv')

    # Parse list strings
    for col in ['oa_refs_oaid', 'ss_refs_doi', 'ss_refs_ssid', 'refs_oaids_from_dois', 'refs_dois_from_oaids']:
        df[col] = df[col].apply(parse_list_string)

    if filter_good_papers:
        mask = get_good_papers_mask(df)
        df = df[mask]

    if filter_good_references:
        mask = get_good_references_mask(ref_df)
        ref_df = ref_df[mask]

    return df, ref_df
