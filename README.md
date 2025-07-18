# Sentence API

OpenAI-compatible API providing **sentence embeddings** and **cross-encoder re-ranking** powered by [Sentence Transformers](https://www.sbert.net/) and [FastAPI](https://fastapi.tiangolo.com/).

---

## Features

| Endpoint                    | Purpose                                                |
| --------------------------- | ------------------------------------------------------ |
| `POST /v1/embeddings`       | Turn text into dense vectors (single string or batch). |
| `POST /v1/rerank`           | Score / reorder candidate documents for a query.       |
| `GET  /v1/embeddings/models`| List available embedding models.                       |
| `GET  /v1/rerank/models`    | List available rerank models.                          |
| `GET  /healthz`             | Liveness probe for your orchestrator.                  |

Both `/v1/*` routes follow the OpenAI API schema so you can drop-in replace OpenAI calls with your own self-hosted inference.

---

## Container image

`ghcr.io/empire-tm/sentence-api` <sub>(or build locally with `docker build -t sentence-api .`)</sub>

---

## Running

### TL;DR (CPU)

```bash
docker run -p 8080:8080 \
  -e EMBED_MODEL_NAMES='["all-MiniLM-L6-v2"]' \
  -e RERANK_MODEL_NAMES='["cross-encoder/ms-marco-MiniLM-L-6-v2"]' \
  ghcr.io/empire-tm/sentence-api:latest
```

### TL;DR (GPU)

If your host has an NVIDIA GPU and the *nvidia-container-toolkit* is installed:

```bash
docker run --gpus all -p 8080:8080 \
  -e EMBED_MODEL_NAMES='["all-MiniLM-L6-v2","paraphrase-mpnet-base-v2"]' \
  -e RERANK_MODEL_NAMES='["cross-encoder/ms-marco-MiniLM-L-6-v2"]' \
  ghcr.io/empire-tm/sentence-api:latest
```

### Docker Compose

Create **docker-compose.yml**:

```yaml
services:
  sentence-api:
    image: ghcr.io/empire-tm/sentence-api:latest
    ports:
      - "8080:8080"
    environment:
      EMBED_MODEL_NAMES: '["all-MiniLM-L6-v2"]'
      RERANK_MODEL_NAMES: '["cross-encoder/ms-marco-MiniLM-L-6-v2"]'
```

Then start the server:

```bash
docker compose up -d
```

---

## Installation

### Docker

```bash
docker run -p 8080:8080 \
  -e EMBED_MODEL_NAMES='["all-MiniLM-L6-v2"]' \
  -e RERANK_MODEL_NAMES='["cross-encoder/ms-marco-MiniLM-L-6-v2"]' \
  ghcr.io/empire-tm/sentence-api
```

### Local (Python ≥ 3.9)

```bash
git clone https://github.com/empire-tm/sentence-api
cd sentence-api
pip install -r requirements.txt
uvicorn main:app --port 8080 --reload
```

---

## Quick start

### Interactive docs

Once running, open **http://localhost:8080/docs** for the Swagger UI.

### Get embeddings

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
        "input": ["Sentence A", "Sentence B"],
        "model": "all-MiniLM-L6-v2"
      }'
```

### List available embedding models

```bash
curl http://localhost:8080/v1/embeddings/models
```

### Re-rank documents

```bash
curl http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What is the air-speed of an unladen swallow?",
        "documents": [
          {"text": "African or European?"},
          {"text": "It depends on the swallow."}
        ],
        "top_n": 1,
        "return_documents": true,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
      }'
```

### List available rerank models

```bash
curl http://localhost:8080/v1/rerank/models
```

The server responds with relevance scores (higher = better).

---

## Environment variables

| Variable             | Default (fallback)                         | Description                                                      |
| --------------------|--------------------------------------------|------------------------------------------------------------------|
| `EMBED_MODEL_NAMES`  | `["all-MiniLM-L6-v2"]`                     | JSON array of embedding model names to preload.                  |
| `RERANK_MODEL_NAMES` | `["cross-encoder/ms-marco-MiniLM-L-6-v2"]` | JSON array of cross-encoder models to preload.                   |

The first model in each list is used as default when no model is specified in the request.

---

## Supported models

Any model compatible with [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) or [CrossEncoder](https://www.sbert.net/docs/pretrained_models.html#cross-encoders) works out-of-the-box.

Popular choices:

**Embedding models**
- `all-MiniLM-L6-v2`
- `multi-qa-MiniLM-L6-cos-v1`
- `paraphrase-mpnet-base-v2`

**Rerank (cross-encoder) models**
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `cross-encoder/qnli-roberta-base`

---

## Acknowledgements

This project was inspired by the **[stapi](https://github.com/substratusai/stapi)** project from SubstratusAI.  
Thanks to the stapi team for the idea and motivation!

---

⭐ **Star this repo if it helped you!**