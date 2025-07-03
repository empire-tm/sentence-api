import json
from typing import Union, List, Dict, Optional
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------- Model registries ---------------- #
embed_models: Dict[str, SentenceTransformer] = {}
rerank_models: Dict[str, CrossEncoder] = {}

EMBED_MODEL_NAMES = json.loads(os.getenv(
    "EMBED_MODEL_NAMES", '["all-MiniLM-L6-v2"]'
))
RERANK_MODEL_NAMES = json.loads(os.getenv(
    "RERANK_MODEL_NAMES", '["cross-encoder/ms-marco-MiniLM-L-6-v2"]'
))

DEFAULT_EMBED_MODEL = EMBED_MODEL_NAMES[0] if EMBED_MODEL_NAMES else None
DEFAULT_RERANK_MODEL = RERANK_MODEL_NAMES[0] if RERANK_MODEL_NAMES else None

# ---------------- Pydantic schemas ---------------- #
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(
        examples=["Cats are wonderful companions"]
    )
    model: str = Field(default=DEFAULT_EMBED_MODEL, examples=EMBED_MODEL_NAMES)


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
    model: str = Field(default=DEFAULT_RERANK_MODEL, examples=RERANK_MODEL_NAMES)


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
    # Embed-модели
    for name in EMBED_MODEL_NAMES:
        if name not in embed_models:
            embed_models[name] = SentenceTransformer(name, trust_remote_code=True)

    # Rerank-модели
    for name in RERANK_MODEL_NAMES:
        if name not in rerank_models:
            rerank_models[name] = CrossEncoder(name)

    yield


app = FastAPI(
        lifespan=lifespan, 
        version="0.0.3", 
        title="Sentence API",
        summary="OpenAI-compatible API providing sentence embeddings and cross-encoder re-ranking powered by Sentence Transformers and FastAPI."
    )


# ---------------- Endpoints ---------------- #
@app.post("/v1/embeddings", tags=["Embeddings"])
async def embedding(item: EmbeddingRequest) -> EmbeddingResponse:
    model_name = item.model
    model = embed_models.get(model_name)
    if model is None:
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


@app.get("/v1/embeddings/models", tags=["Embeddings"])
async def get_embeddings_models() -> list[str]:
    return EMBED_MODEL_NAMES


@app.post("/v1/rerank", tags=["Rerank"])
async def rerank(item: RerankRequest) -> RerankResponse:
    model_name = item.model
    model = rerank_models.get(model_name)
    if model is None:
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

@app.get("/v1/rerank/models", tags=["Rerank"])
async def get_rerank_models() -> list[str]:
    return RERANK_MODEL_NAMES

@app.get("/", tags=["System"])
@app.get("/healthz", tags=["System"])
async def healthz():
    return {"status": "ok"}