from semantic_search.utils import load_metadata
from semantic_search.store.models import LocalEmbeddingModel
from semantic_search.store.milvus_store import MilvusDocumentStore
from semantic_search.store.store import FAISSDocumentStore

df, ref_df = load_metadata(
    '/Users/luis/Desktop/ETH/Courses/SS25-DSL/raw-data/metadata3',
    filter_good_papers=True,
    filter_good_references=True
)
ref_df.rename(columns={'oaid': 'id', 'abstract': 'text'}, inplace=True)

# ds = FAISSDocumentStore(
#     embedding_model=LocalEmbeddingModel(),
#     db_superdir='/Users/luis/Desktop/ETH/Courses/SS25-DSL/db',
#     store_name='test1',   
# )

ds = MilvusDocumentStore(
    embedding_model=LocalEmbeddingModel(), 
    db_superdir='/Users/luis/Desktop/ETH/Courses/SS25-DSL/db',
    store_name='milvus_dev3'
)

ds.create_index_from_df(ref_df.rename(columns={'oaid': 'id', 'abstract': 'text'}).iloc[:100], overwrite=True)

print(ds.search('Attention is all you need', top_k=100, return_scores=True, return_doc_metadata=False, retrieval_method='hybrid'))