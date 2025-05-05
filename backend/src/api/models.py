from pydantic import BaseModel
from typing import List, Optional

class TextRequest(BaseModel):
    text: str

class Citation(BaseModel):
    id: int
    title: str
    author: str
    year: int
    publisher: str
    relevance: int
    citation: str

class CitationsResponse(BaseModel):
    citations: List[Citation]

class Continuation(BaseModel):
    id: int
    text: str
    confidence: int

class ContinuationsResponse(BaseModel):
    continuations: List[Continuation]