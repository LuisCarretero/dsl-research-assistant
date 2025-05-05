from fastapi import FastAPI
from src.api.endpoints import captions, citations, continuations
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DSL Research Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(captions.router, prefix="/api")
app.include_router(citations.router, prefix="/api")
app.include_router(continuations.router, prefix="/api")
