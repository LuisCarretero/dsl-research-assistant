from semantic_search.utils import load_data
from semantic_search.store.models import LocalEmbeddingModel
from semantic_search.store.milvus_store import MilvusDocumentStore
from semantic_search.store.faiss_store import FAISSDocumentStore
from time import sleep
import numpy as np

df, ref_df = load_data(
    '/Users/luis/Desktop/ETH/Courses/SS25-DSL/raw-data/metadata3',
    filter_good_papers=True,
    filter_good_references=True
)
ref_df.rename(columns={'oaid': 'id', 'abstract': 'text'}, inplace=True)

print('Creating store...')
ds = FAISSDocumentStore(
    embedding_model=LocalEmbeddingModel(),
    db_superdir='/Users/luis/Desktop/ETH/Courses/SS25-DSL/db',
    store_name='test1',   
)

# ds = MilvusDocumentStore(
#     embedding_model=LocalEmbeddingModel(), 
#     db_superdir='/Users/luis/Desktop/ETH/Courses/SS25-DSL/db',
#     store_name='milvus_dev3'
# )

print('Creating index...')
ds.load_store(allow_fail=False)
# ds.create_index_from_df(ref_df.rename(columns={'oaid': 'id', 'abstract': 'text'}).iloc[:100], overwrite=True)
sleep(2)

print('Searching...')

print(ds.index.search(np.random.rand(1, ds.embedding_model.embedding_dim), 10))

# print(ds.search('Attention is all you need', top_k=100, return_scores=True, return_doc_metadata=False, retrieval_method='embedding'))