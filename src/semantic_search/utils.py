import pandas as pd
from pathlib import Path
import re

def parse_referenced_works(x):
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