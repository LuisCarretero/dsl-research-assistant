from semantic_search.utils import collect_orig_paper_metadata, collect_ref_metadata, update_orig_ref_metadata
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True)
    parser.add_argument('--metadata_dir', type=str, required=True)
    parser.add_argument('--max_papers', type=int, default=-1)
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
    update_orig_ref_metadata(orig_output_fpath, refs_output_fpath)
