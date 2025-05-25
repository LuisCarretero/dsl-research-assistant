from pathlib import Path
import argparse


from src.semantic_search.data_retrieval.utils import collect_orig_paper_metadata, collect_ref_metadata, update_orig_ref_metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='/Users/luis/Desktop/ETH/Courses/SS25-DSL/dsl-research-assistant/data/Conversions/opencvf-data/txt')
    parser.add_argument('--metadata_dir', type=str, default='/Users/luis/Desktop/ETH/Courses/SS25-DSL/dsl-research-assistant/data/metadata3')
    parser.add_argument('--max_papers', type=int, default=-1)
    parser.add_argument('--skip_orig_metadata_update', action='store_true')
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    metadata_dir = Path(args.metadata_dir)

    orig_output_fpath = metadata_dir / 'orig.csv'
    refs_output_fpath = metadata_dir / 'refs.csv'

    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Collect metadata for original papers
    if not orig_output_fpath.exists():
        collect_orig_paper_metadata(raw_dir, orig_output_fpath, max_papers=args.max_papers)
    else:
        print(f"Not collecting original paper metadata because it already exists at {orig_output_fpath}")

    # Collect metadata for references
    if not refs_output_fpath.exists():
        collect_ref_metadata(orig_output_fpath, refs_output_fpath)
    else:
        print(f"Not collecting reference metadata because it already exists at {refs_output_fpath}")

    # Update original paper metadata with reference metadata from other source
    if not args.skip_orig_metadata_update:
        update_orig_ref_metadata(orig_output_fpath, refs_output_fpath)
    else:
        print(f"Not updating original paper metadata with reference metadata because it already exists at {orig_output_fpath}")

# poetry run python -m src.semantic_search.data_retrieval.retrieve_paper_metadata --skip_orig_metadata_update