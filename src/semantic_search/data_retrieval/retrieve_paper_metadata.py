from semantic_search.utils import collect_orig_paper_metadata, collect_ref_metadata
import os

if __name__ == '__main__':
    # Collect metadata for original papers
    raw_dir = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/Conversions/opencvf-data/txt'
    orig_output_fpath = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata/orig.csv'
    if not os.path.exists(orig_output_fpath):
        collect_orig_paper_metadata(raw_dir, orig_output_fpath, max_papers=-1)

    # Collect metadata for references
    refs_output_fpath = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata/refs.csv'
    if not os.path.exists(refs_output_fpath):
        collect_ref_metadata(orig_output_fpath, refs_output_fpath)
