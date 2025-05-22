import os
import json
import pandas as pd
from pymilvus import MilvusClient

from semantic_search.utils import load_data

def update_store_doc_metadata(
    store_name: str, 
    db_superdir: str = '/Users/luis/Desktop/ETH/Courses/SS25-DSL/db', 
    metadata_dirpath: str = '/Users/luis/Desktop/ETH/Courses/SS25-DSL/data/metadata3'
):
    """
    Updates the store document data with new columns from the reference metadata .csv file at 
    metadata_dirpath. Useful when scraping new data and wanting to add it to DocumentStore without
    having to reinitialize it.
    """
    # Load new metadata 
    _, ref_df = load_data(
        metadata_dirpath,
        filter_good_papers=True,
        filter_good_references=True
    )
    ref_df.rename(columns={'oaid': 'id', 'abstract': 'text'}, inplace=True)

    # Load store document data
    doc_store_path = os.path.join(db_superdir, store_name, 'documents.parquet')
    docs = pd.read_parquet(doc_store_path)

    # Make sure we are operating on the same data
    if not set(docs['id'].values) == set(ref_df['id'].values):
        print(f'Warning: Difference in ids: {set(docs["id"].values) - set(ref_df["id"].values)} are in database DF but not in metadata DF.')

    # Get columns that are in ref_df but not in docs
    new_columns = [col for col in ref_df.columns if col not in docs.columns]
    for col in new_columns:
        # Use the id column to map values from ref_df to docs
        mapping = dict(zip(ref_df['id'], ref_df[col]))
        # Add the new column to docs
        docs[col] = docs['id'].map(mapping)   
    print(f"Added {len(new_columns)} new columns: {new_columns}")

    docs.to_parquet(doc_store_path)

def get_orphaned_milvus_collections(db_superdir: str, milvus_uri: str = 'http://localhost:19530', print_collections: bool = True) -> list[str]:
    """
    Clean Milvus database by dropping all collections that are not associated with a database 
    directory and accompanying metadata file.

    Usage: 

        from pymilvus import MilvusClient
        
        client = MilvusClient()
        for coll_name in get_orphaned_milvus_collections(db_superdir='/Users/luis/Desktop/ETH/Courses/SS25-DSL/db', print_collections=False):
            client.drop_collection(coll_name)

    """
    # Get all metadata files # Files are in db superdir under some_dir/metadata.json ->  store -> storename
    # Find all store directories in the db_superdir
    store_names = []
    for store_dir in os.listdir(db_superdir):
        metadata_path = os.path.join(db_superdir, store_dir, 'metadata.json')
        if os.path.isdir(os.path.join(db_superdir, store_dir)) and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if 'store' in metadata and 'store_name' in metadata['store']:
                        store_names.append(metadata['store']['store_name'])
            except (json.JSONDecodeError, IOError):
                # Skip if metadata file is invalid
                continue
    
    # Print collections that are not associated with a metadata file
    client = MilvusClient(uri=milvus_uri)
    collections_in_milvus = client.list_collections()
    # Find collections that exist in Milvus but not in our store directories
    missing_collections = [coll for coll in collections_in_milvus if coll not in store_names]

    if print_collections:
        print(f"Store names found in {db_superdir}: {store_names}")
        print(f"Collections found in Milvus: {collections_in_milvus}")
        
        if missing_collections:
            print(f"Collections that could be dropped (not associated with metadata): {missing_collections}")
            print("To drop these collections, use: client.drop_collection(collection_name)")
        else:
            print("No orphaned collections found.")

    return missing_collections
