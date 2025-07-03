from typing import Union, List, Dict, Optional
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------- Model registries ---------------- #
embed_models: Dict[str, SentenceTransformer] = {}
rerank_models: Dict[str, CrossEncoder] = {}

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- Pydantic schemas ---------------- #
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(
        examples=["Cats are wonderful companions"]
    )
    model: str = Field(default=EMBED_MODEL_NAME, examples=[EMBED_MODEL_NAME])


class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int
    object: str = "embedding"


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    usage: Usage
    object: str = "list"


# ---- Rerank ---- #
class RerankDocument(BaseModel):
    """Document container. For now we only support plain text."""

    text: str


class RerankRequest(BaseModel):
    query: str = Field(
        examples=["Why do cats purr?"]
    )
    documents: List[RerankDocument] = Field(
        examples=[
            [
                RerankDocument(text="Cats often purr when they feel relaxed and happy."),
                RerankDocument(text="Purring can also serve as a self-healing mechanism."),
                RerankDocument(text="Dogs bark to communicate with humans and other dogs.")
            ]
        ]
    )
    top_n: Optional[int] = None
    return_documents: bool = False
    model: str = Field(default=RERANK_MODEL_NAME, examples=[RERANK_MODEL_NAME])


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[RerankDocument] = None


class RerankResponse(BaseModel):
    model: str
    usage: Usage
    results: List[RerankResult]


# ---------------- App lifecycle ---------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once during startup
    if len(EMBED_MODEL_NAME) != 0:
        embed_models[EMBED_MODEL_NAME] = SentenceTransformer(
            EMBED_MODEL_NAME, trust_remote_code=True
        )
    if len(RERANK_MODEL_NAME) != 0:
        rerank_models[RERANK_MODEL_NAME] = CrossEncoder(RERANK_MODEL_NAME)
    yield


app = FastAPI(
        lifespan=lifespan, 
        version="0.0.2", 
        title="Sentence API",
        summary="OpenAI-compatible API providing sentence embeddings and cross-encoder re-ranking powered by Sentence Transformers and FastAPI."
    )


# ---------------- Endpoints ---------------- #
@app.post("/v1/embeddings")
async def embedding(item: EmbeddingRequest) -> EmbeddingResponse:
    model_name = item.model
    model = embed_models.get(model_name)
    if model is None or len(model_name) == 0:
        raise HTTPException(status_code=404, detail=f"embedding model '{model_name}' not loaded")

    def _encode(text: str) -> List[float]:
        return model.encode(text).tolist()

    if isinstance(item.input, str):
        vec = _encode(item.input)
        tokens = len(vec)
        return EmbeddingResponse(
            data=[EmbeddingData(embedding=vec, index=0)],
            model=model_name,
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
        )

    if isinstance(item.input, list):
        embeddings = []
        tokens = 0
        for idx, text in enumerate(item.input):
            if not isinstance(text, str):
                raise HTTPException(status_code=400, detail="input must be string or list[str]")
            vec = _encode(text)
            tokens += len(vec)
            embeddings.append(EmbeddingData(embedding=vec, index=idx))
        return EmbeddingResponse(
            data=embeddings,
            model=model_name,
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
        )

    raise HTTPException(status_code=400, detail="input must be string or list[str]")


@app.post("/v1/rerank")
async def rerank(item: RerankRequest) -> RerankResponse:
    model_name = item.model
    model = rerank_models.get(model_name)
    if model is None or len(model_name) == 0:
        raise HTTPException(status_code=404, detail=f"rerank model '{model_name}' not loaded")

    if not item.documents:
        raise HTTPException(status_code=400, detail="documents list cannot be empty")

    # Prepare sentence pairs
    pairs = [(item.query, doc.text) for doc in item.documents]
    scores = model.predict(pairs).tolist()

    # Build ranking
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    if item.top_n:
        ranked = ranked[: item.top_n]

    results: List[RerankResult] = []
    total_tokens = 0

    for idx, score in ranked:
        result = RerankResult(index=idx, relevance_score=float(score))
        if item.return_documents:
            result.document = item.documents[idx]
        results.append(result)

        # Rough token counting (query + doc)
        total_tokens += len(model.tokenizer.encode(item.query)) + len(
            model.tokenizer.encode(item.documents[idx].text)
        )

    usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)
    return RerankResponse(model=model_name, usage=usage, results=results)


@app.get("/")
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}