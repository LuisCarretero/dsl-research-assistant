from llama_index import GPTVectorStoreIndex

def load_index(index_path="papers_index.json"):
    index = GPTSimpleVectorIndex.load_from_disk(index_path)
    return index

def search_for_citations(query, index):
    response = index.query(query)
    return response