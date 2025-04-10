from semanticscholar import SemanticScholar
from semanticscholar.SemanticScholarException import ObjectNotFoundException
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re
from pathlib import Path
import pandas as pd
from functools import partial


def get_title(fpath: str):
    doc_text = Path(fpath).read_text(encoding="utf-8")
    title_match = re.search(r'## ([^\n#]+)', doc_text)
    if title_match is None:  # Fallback: Extract from file name
        return ' '.join(Path(fpath).stem.split('_')[1:-3])
    return title_match.group(1)


def get_paper_metadata(sch: SemanticScholar, paper_title: str):
    try:
        paper = sch.search_paper(query=paper_title, match_title=True, fields=['paperId', 'externalIds', 'abstract'])
        res = {'paperId': paper['paperId'], 'abstract': paper['abstract']}
        res.update({f'externalIds.{k}': v for k, v in paper['externalIds'].items()})
        return res
    except ObjectNotFoundException:
        return {}

if __name__ == '__main__':
    raw_dir = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/Conversions/opencvf-data/txt'
    output_dir = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata'
    
    # Get paths, fnames and titles from raw data dir
    df = pd.DataFrame(
        [(str(fpath), fpath.name) for fpath in Path(raw_dir).glob("*.txt")], 
        columns=['fpath', 'fname']
    )
    df['title'] = df['fpath'].apply(get_title)
    
    # Get paper metadata from Semantic Scholar
    sch = SemanticScholar()
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(
            executor.map(partial(get_paper_metadata, sch), df['title'].values),
            total=len(df)
        ))

    # Concat and save
    df = pd.concat([df, pd.DataFrame(results)], axis=1)
    df.to_csv(Path(output_dir, 'paper_metadata.csv'), index=False)
