from fastapi import APIRouter
from src.api.models import TextRequest, CitationsResponse, Citation
from src.semantic_search.store.milvus_store import MilvusDocumentStore, MilvusIndexNotAvailableError

router = APIRouter()


ds = MilvusDocumentStore(db_superdir='/Users/luis/Desktop/ETH/Courses/SS25-DSL/dsl-research-assistant/db', store_name='main')
try:
    ds.load_store()
except MilvusIndexNotAvailableError as e:
    print(f"Could not load Milvus index. Trying to rebuild it...")
    ds.rebuild_index_from_dir(overwrite=False, allow_embedding_calc=True)
    ds.load_store()


@router.post("/generate-citations/", response_model=CitationsResponse)
async def generate_citations(request: TextRequest):
    """
    Generate citation recommendations based on the provided text.
    """
    query = request.text
    results = ds.search(query, top_k=3, return_doc_metadata=True)


    # Convert search results to Citation objects
    citations = []
    for i, result in enumerate(results):
        # Create a citation object from each result
        citation = Citation(
            id=result.get('rank', ''),
            title=result.get('title', ''),
            author=result.get('authors', ''),
            year=int(result.get('pub_date', '').split('-')[0]) if result.get('pub_date') else None,
            publisher=result.get('publisher', ''),
            relevance=int(result.get('score', 0) * 100),  # Convert score to percentage
            citation=result.get('cit_str', '')
        )
        citations.append(citation)  
    
    # Return early if we have results from the search
    if citations:
        return CitationsResponse(citations=citations)
