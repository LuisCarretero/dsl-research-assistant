from fastapi import APIRouter
from src.api.models import TextRequest, CitationsResponse, Citation


router = APIRouter()

@router.post("/generate-citations/", response_model=CitationsResponse)
async def generate_citations(request: TextRequest):
    """
    Generate citation recommendations based on the provided text.
    """
    citations = [
        Citation(
            id=1,
            title="Machine Learning: A Probabilistic Perspective",
            author="Kevin P. Murphy",
            year=2012,
            publisher="MIT Press",
            relevance=92,
            citation="Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press."
        ),
        Citation(
            id=2,
            title="Deep Learning",
            author="Ian Goodfellow, Yoshua Bengio, Aaron Courville",
            year=2016,
            publisher="MIT Press",
            relevance=87,
            citation="Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press."
        ),
        Citation(
            id=3,
            title="Artificial Intelligence: A Modern Approach",
            author="Stuart Russell, Peter Norvig",
            year=2020,
            publisher="Pearson",
            relevance=78,
            citation="Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach. Pearson."
        )
    ]
    
    return CitationsResponse(citations=citations)