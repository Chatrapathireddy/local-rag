from threading import RLock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import RAGPipeline

app = FastAPI(title="Local RAG API", version="1.0.0")
_pipeline: RAGPipeline | None = None
_pipeline_lock = RLock()


class IngestRequest(BaseModel):
    documents: dict[str, str] = Field(min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    min_score: float = Field(default=0.2, ge=0.0, le=1.0)


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            embedding_model_name="all-MiniLM-L6-v2",
            generator_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
            chunk_size=400,
            chunk_overlap=80,
        )
    return _pipeline


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, int]:
    documents = {
        name.strip(): text
        for name, text in request.documents.items()
        if name.strip() and text.strip()
    }
    if not documents:
        raise HTTPException(status_code=422, detail="Provide at least one non-empty document.")

    with _pipeline_lock:
        chunks = get_pipeline().ingest_texts(documents)
    return {"documents": len(documents), "chunks": chunks}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank.")

    with _pipeline_lock:
        return get_pipeline().query(
            question,
            top_k=request.top_k,
            min_score=request.min_score,
        )