import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Any, Iterable, Callable, Tuple
import pyalex
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from semanticscholar import SemanticScholar
from functools import partial
from difflib import SequenceMatcher


pyalex.config.email = "luis.carretero@gmx.de"
pyalex.config.max_retries = 10
pyalex.config.retry_backoff_factor = 0.1
pyalex.config.retry_http_codes = [429, 500, 503]


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

def extract_abstract_from_md(fpath: str) -> str:
    """
    Extract the abstract from a markdown file created by Docling.
    """
    doc_text = Path(fpath).read_text(encoding="utf-8")
    abstract_match = re.search(r'## Abstract\n\n(.*?)(?=\n\n## \d+\.)', doc_text, re.DOTALL)
    return abstract_match.group(1) if abstract_match else ''

def get_title_from_fpath(fpath: str) -> str:
    doc_text = Path(fpath).read_text(encoding="utf-8")
    title_match = re.search(r'## ([^\n#]+)', doc_text)

    if title_match:
        return title_match.group(1)

    # Fallback
    return ' '.join(fpath.split('.')[0].split('_')[1:-3])

def similarity_ratio(a: str, b: str) -> float:
    """
    Calculate the similarity ratio between two strings.
    Returns a float between 0 and 1, where 1 means identical strings.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_orig_metadata_oa(title: str, reraise: bool = False) -> Dict[str, Any]:
    """
    Retrieves metadata associated with original paper (as opposed to secondary references) via OpenAlex API (oa).

    FIXME: Could call batched once we have ID.
    """
    try:
        # Pull top 20 matches and sort by title similarity match score (OpenAlex somehow doesn't do this?)
        res = pyalex.Works().search_filter(title=title).select(['id', 'title']).get(page=1, per_page=20)
        for paper_dict in res:
            paper_dict['similarity'] = similarity_ratio(paper_dict['title'], title)
        res = sorted(res, key=lambda x: x['similarity'], reverse=True)

        sim_score_1st, sim_score_2nd = res[0]['similarity'], (res[1]['similarity'] if len(res) > 1 else 0)

        search_results = pyalex.Works().filter_or(openalex_id=res[0]['id']).select(['id', 'doi', 'title', 'referenced_works']).get(page=1, per_page=1)
        if len(search_results) == 1:
            res = search_results[0]

            # Extract ID: https://openalex.org/W4402916217 -> W4402916217
            refs = list(map(lambda x: x.split('/')[-1], res.get('referenced_works', [])))
            
            return {
                'oaid': res.get('id').split('/')[-1],
                'oa_doi': res.get('doi').split('https://doi.org/')[-1],
                'oa_sim_score_1st': sim_score_1st,
                'oa_sim_score_2nd': sim_score_2nd,
                'oa_ref_cnt': len(refs),
                'oa_refs_oaid': refs
            }
        else:
            print(f"Warning: Found {len(search_results)} results for \"{title}\" with OAID {res[0]['id']}")
    except Exception as e:
        if reraise: raise e
        print(f"Error fetching OA metadata for \"{title}\": {e}")
    return {}

def get_orig_metadata_ss(
    sch: SemanticScholar, 
    title: str, 
    use_ref_query: bool = False, 
    reraise: bool = False
) -> Dict[str, Any]:
    """
    TODO: Combine into single API call if possible?
    Problem with above: SS paper class only has reference title and SSID so we would 
    need another call to get reference external IDs.

    # TODO: search_paper() can throw ObjectNotFoundException. Handle this seperately?
    from semanticscholar.SemanticScholarException import ObjectNotFoundException

    # TODO: Think about what to do with references without ID? 
    # We have len(refs_ssid) < ref_cnt in most cases due to this.
    """
       
    try:
        # Search SS paper by title
        raw = sch.search_paper(title, fields=['paperId', 'externalIds', 'title', 'referenceCount', 'references'], match_title=True)
        ssid, title_ss, ref_cnt = raw['paperId'], raw['title'], raw['referenceCount']
        doi = raw['externalIds'].get('DOI') if raw['externalIds'] else None
        title_sim = similarity_ratio(title_ss, title)
        refs_ssid = [x.get('paperId') for x in raw['references'] if x.get('paperId') is not None]

        if use_ref_query:
            refs_raw = map(
                lambda x: x['citedPaper'], 
                sch.get_paper_references(paper_id=ssid, fields=['externalIds'], limit=1000).items
            )
        else:
            refs_raw = sch.get_papers(paper_ids=refs_ssid, fields=['externalIds'])

        # Parse external reference ids
        refs_doi = []
        for item in refs_raw:
            external_ids = item['externalIds']
            if external_ids is None: continue
            ref_doi = external_ids.get('DOI')
            if ref_doi is None: 
                ref_arxiv = external_ids.get('ArXiv')
                if ref_arxiv is None: continue
                ref_doi = f'10.48550/arXiv.{ref_arxiv}'
            refs_doi.append(ref_doi)

        return {
            'ssid': ssid,
            'ss_doi': doi,
            'ss_sim_score': title_sim,
            'ss_ref_cnt': ref_cnt,
            'ss_refs_ssid': refs_ssid,
            'ss_refs_doi': refs_doi
        }
    except Exception as e:
        if reraise: raise e
        print(f"Error fetching SS metadata for title: \"{title}\": {e}")
    return {}

def multithread_apply(data: Iterable, func: Callable, n_workers: int = 5, progress_bar: bool = True, desc=None) -> List[Any]:
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(func, data),
            total=len(data),
            disable=not progress_bar,
            desc=desc
        ))
    return results

def count_references(row: pd.Series, df: pd.DataFrame) -> Tuple[int, int]:
    total_refs = len(row['referenced_works'])
    dataset_oaids = set(df['oaid'].dropna())
    refs_in_dataset = sum(1 for ref in row['referenced_works'] if ref in dataset_oaids)
    return total_refs, refs_in_dataset

def uninvert_abstract(inv_index: Dict[str, List[int]]) -> str:
    l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
    return ' '.join(map(lambda x: x[0], sorted(l_inv, key=lambda x: x[1])))

def get_ref_metadata_oa(ref_ids: List[str], id_key: str = 'openalex_id', progress_bar: bool = False) -> np.ndarray:
    """
    Get metadata of interest for each reference work using OpenAlex API. Searches by OpenAlex ID.
    """
    fields_of_interest = ['id', 'doi','abstract_inverted_index', 'title', 'type', 'topics']

    if len(ref_ids) == 0: return np.array([])
    if id_key == 'openalex_id':
        # OpenAlex IDs only to reduce request line size (may get bad request if too long)
        ref_ids = list(map(lambda x: x.split('/')[-1], ref_ids))

    res = []
    batch_size = 100  # Max PyAlex limit
    batch_cnt = (len(ref_ids)-1) // batch_size + 1

    for i in tqdm(range(batch_cnt), disable=not progress_bar):
        batch = ref_ids[i*batch_size:(i+1)*batch_size]

        raw = pyalex.Works().filter_or(**{id_key: batch}).select(fields_of_interest).get(per_page=len(batch))
        # if len(raw) != len(batch):  # FIXME: Check how many are actually missing
        #     print(f"Warning: Only {len(raw)} out of {len(batch)} references found for batch {i}")

        for item in raw:
            if item['abstract_inverted_index'] is not None:
                abstract = uninvert_abstract(item['abstract_inverted_index'])
            else:
                abstract = ''
            res.append({
                'oaid': item['id'].split('/')[-1] if item['id'] else None,
                'doi': item['doi'].split('https://doi.org/')[-1] if item['doi'] else None,
                'ref_via': id_key,
                'title': item.get('title'),
                'abstract': abstract, 
                'type': item.get('type'), 
                'topic': item['topics'][0]['display_name'] if item['topics'] else None,
                'domain': item['topics'][0]['domain']['display_name'] if item['topics'] else None,
                'field': item['topics'][0]['field']['display_name'] if item['topics'] else None,
                'subfield': item['topics'][0]['subfield']['display_name'] if item['topics'] else None
            })
    return np.array(res)

def collect_orig_paper_metadata(raw_dir: str, output_fpath: str, max_papers: int = -1) -> None:
    """
    Collect metadata of original papers from Docling output directory.
    """
    # Get fpath, fname
    df = pd.DataFrame([(str(fpath), fpath.name) for fpath in Path(raw_dir).glob("*.txt")], columns=['fpath', 'fname'])
    df = df.iloc[:max_papers]
    df['title'] = df['fpath'].apply(get_title_from_fpath)

    # Load OpenAlex metadata
    oa_metadata = pd.DataFrame(multithread_apply(
        df['title'].values, 
        get_orig_metadata_oa, 
        n_workers=4, 
        desc='Pulling OpenAlex metadata'
    ))
    assert len(oa_metadata) == len(df)
    df = pd.concat([df, oa_metadata], axis=1)

    # Load SemanticScholar metadata
    sch = SemanticScholar()  # FIXME: Implement exp backoff and remove multithreading
    ss_metadata = pd.DataFrame(multithread_apply(
        df['title'].values, 
        partial(get_orig_metadata_ss, sch), 
        n_workers=1, 
        desc='Pulling SemanticScholar metadata'
    ))
    assert len(ss_metadata) == len(df)
    df = pd.concat([df, ss_metadata], axis=1)

    df.to_csv(output_fpath, index=False)

def collect_ref_metadata(orig_metadata_fpath: str, output_fpath: str, max_papers: int = -1) -> None:
    """
    Collect metadata of references from specified by original paper metadata.
    """
    # Load original paper metadata
    df = pd.read_csv(orig_metadata_fpath)
    df = df.iloc[:max_papers]
    df['oa_refs_oaid'] = df['oa_refs_oaid'].apply(parse_list_string)
    df['ss_refs_doi'] = df['ss_refs_doi'].apply(parse_list_string)

    # Retrieve reference metadata via OpenAlex API (using both DOI and OAID)
    results = []
    for col, id_key, domain in zip(['oa_refs_oaid', 'ss_refs_doi'], ['openalex_id', 'doi'], ['openalex', 'doi']):
        # Collects all refs
        all_refs = pd.Series(np.concatenate(df[col].values)).str.split(f'https://{domain}.org/').str[-1].unique().tolist()
        # Batch into to minimize API calls
        all_refs_batched = [all_refs[i:i+100] for i in range(0, len(all_refs), 100)]
        # Call API multithreaded
        results.extend(multithread_apply(
            all_refs_batched, 
            lambda x: get_ref_metadata_oa(x, id_key=id_key, progress_bar=False), 
            n_workers=5, 
            desc=f'Pulling {domain} reference metadata via OpenAlex',
            progress_bar=True
        ))
    results = np.concatenate([res for res in results if len(res) > 0])

    ref_df = pd.DataFrame(results.tolist())
    ref_df = ref_df.drop_duplicates(subset=['oaid', 'doi'])  # Checked manually and in almost all (all but 10/30000 cases) the data agrees

    ref_df.to_csv(output_fpath, index=False)

def convert_and_compare_refs(oaids: List[str], dois: List[str], ref_df: pd.DataFrame) -> Dict[str, Any]:
    oaids = set(map(lambda x: x.lower(), oaids))
    dois = set(map(lambda x: x.lower(), dois))

    dois_from_oaids = ref_df.doi.str.lower()[ref_df.oaid.str.lower().isin(oaids)].tolist()
    oaids_from_dois = ref_df.oaid.str.lower()[ref_df.doi.str.lower().isin(dois)].tolist()
    res = {
        'refs_oaids_from_dois': oaids_from_dois, 
        'refs_dois_from_oaids': dois_from_oaids,
    }

    # Calculate Jaccard index (intersection over union)
    if len(oaids_from_dois) > 0 and len(dois_from_oaids) > 0:
        oaids_from_dois, dois_from_oaids = set(oaids_from_dois), set(dois_from_oaids)
        iou_oaids = len(oaids_from_dois.intersection(oaids)) / len(oaids_from_dois.union(oaids))
        iou_dois = len(dois_from_oaids.intersection(dois)) / len(dois_from_oaids.union(dois))
        res['ref_jaccard'] = (iou_oaids + iou_dois) / 2
    else:
        res['ref_jaccard'] = None

    return res

def update_orig_ref_metadata(orig_metadata_fpath: str, ref_metadata_fpath: str) -> None:
    """
    Update original paper metadata with reference metadata from different sources.
    """
    df = pd.read_csv(orig_metadata_fpath)
    df['oa_refs_oaid'] = df['oa_refs_oaid'].apply(parse_list_string)
    df['ss_refs_doi'] = df['ss_refs_doi'].apply(parse_list_string)
    ref_df = pd.read_csv(ref_metadata_fpath)

    # Convert and compare refs
    print(f"Converting and comparing references")
    res = df.apply(lambda row: convert_and_compare_refs(row.oa_refs_oaid, row.ss_refs_doi, ref_df), axis=1)
    df = pd.concat([df, pd.DataFrame(res.tolist())], axis=1)

    df.to_csv(orig_metadata_fpath, index=False)
