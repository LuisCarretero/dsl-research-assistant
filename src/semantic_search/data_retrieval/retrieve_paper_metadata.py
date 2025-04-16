from semantic_search.utils import collect_orig_paper_metadata, collect_ref_metadata

if __name__ == '__main__':
    raw_dir = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/Conversions/opencvf-data/txt'
    output_dir = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata'
    
    # Collect metadata for original papers
    raw_dir = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/Conversions/opencvf-data/txt'
    output_fpath = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata/orig.csv'
    collect_orig_paper_metadata(raw_dir, output_fpath, max_papers=-1)

    # Collect metadata for references
    orig_metadata_fpath = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata/orig.csv'
    output_fpath = '/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata/refs.csv'
    collect_ref_metadata(orig_metadata_fpath, output_fpath)
