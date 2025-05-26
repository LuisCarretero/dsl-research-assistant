from fastapi import APIRouter
from src.api.models import TextRequest, CitationsResponse, Citation
from src.semantic_search.store.milvus_store import MilvusDocumentStore, MilvusIndexNotAvailableError
from dotenv import load_dotenv
import os

load_dotenv()

router = APIRouter()

DB_SUPERDIR = os.environ.get("DP_SUPERDIR")

print(DB_SUPERDIR)

ds = MilvusDocumentStore(db_superdir=DB_SUPERDIR, store_name='main')
try:
    ds.load_store(db_superdir=DB_SUPERDIR, store_name='main')
except MilvusIndexNotAvailableError as e:
    print(f"Could not load Milvus index. Trying to rebuild it...")
    ds = MilvusDocumentStore(db_superdir=DB_SUPERDIR, store_name='main')
    ds.rebuild_index_from_dir(db_superdir=DB_SUPERDIR, store_name='main', overwrite=False, allow_embedding_calc=True)
    ds.load_store(db_superdir=DB_SUPERDIR, store_name='main')
results = ds.search("LLMs for literature review generation", top_k=3, return_doc_metadata=True)
print(results)

@router.post("/generate-citations/", response_model=CitationsResponse)
async def generate_citations(request: TextRequest):
    """
    Generate citation recommendations based on the provided text.
    """

    abstract_title = "[Abstract]"
    related_work_title = "[Related Work]"

    abstract = request.text[request.text.find(related_work_title)].replace(abstract_title, "").strip("\n ")

    query = request.text
    results = ds.search(query, top_k=20, return_doc_metadata=True)

    # Convert search results to Citation objects
    citations = []
    for i, result in enumerate(results):
        # Create a citation object from each result
        citation = Citation(
            id=int(result.get('rank', '')) if result.get('rank') else -1,
            title=result.get('title', '') if result.get("title") else "",
            author=result.get('authors', '') if result.get("authors") else "",
            year=int(result.get('pub_date', '').split('-')[0]) if result.get('pub_date') else -1,
            publisher=result.get('publisher', '') if result.get("publisher") else "",
            relevance=int(result.get('score', 0) * 100) if result.get("score") else -1,  # Convert score to percentage
            citation="{" + result.get('cit_str', '') + "}" if result.get("cit_str") else ""
        )
        citations.append(citation)  
    
    # Return early if we have results from the search
    if citations:
        return CitationsResponse(citations=citations)
