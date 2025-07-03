from fastapi.testclient import TestClient
from main import app, EMBED_MODEL_NAMES, RERANK_MODEL_NAMES

# ---------------------- Base checks ---------------------- #

def test_read_healthz():
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

# ---------------------- Embeddings ---------------------- #

def test_embedding_models_list():
    with TestClient(app) as client:
        response = client.get("/v1/embeddings/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        assert set(models) == set(EMBED_MODEL_NAMES)

def test_embedding_str_each_model():
    with TestClient(app) as client:
        for model_name in EMBED_MODEL_NAMES:
            response = client.post("/v1/embeddings", json={
                "input": "Cats purr to express comfort.",
                "model": model_name
            })
            assert response.status_code == 200
            out = response.json()
            assert out["model"] == model_name
            assert isinstance(out["data"][0]["embedding"], list)

def test_embedding_list_str():
    with TestClient(app) as client:
        model_name = EMBED_MODEL_NAMES[0]
        response = client.post("/v1/embeddings", json={
            "input": [
                "Cats often purr when they feel relaxed and happy.",
                "Purring can also serve as a self-healing mechanism.",
            ],
            "model": model_name,
        })
        assert response.status_code == 200
        out = response.json()
        data = out["data"]
        assert len(data) == 2
        assert data[0]["embedding"] != data[1]["embedding"]

# ---------------------- Rerank ---------------------- #

def test_rerank_models_list():
    with TestClient(app) as client:
        response = client.get("/v1/rerank/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        assert set(models) == set(RERANK_MODEL_NAMES)

def test_rerank_basic_each_model():
    with TestClient(app) as client:
        for model_name in RERANK_MODEL_NAMES:
            response = client.post("/v1/rerank", json={
                "query": "Why do cats purr?",
                "documents": [
                    {"text": "Cats purr when content."},
                    {"text": "They also purr to self-heal."},
                    {"text": "Dogs bark for other reasons."}
                ],
                "model": model_name,
            })
            assert response.status_code == 200
            out = response.json()
            assert out["model"] == model_name
            scores = [r["relevance_score"] for r in out["results"]]
            assert scores == sorted(scores, reverse=True)

def test_rerank_top_n():
    with TestClient(app) as client:
        model_name = RERANK_MODEL_NAMES[0]
        response = client.post("/v1/rerank", json={
            "query": "Why do cats purr?",
            "documents": [
                {"text": "Cats often purr when they feel relaxed and happy."},
                {"text": "Purring can also serve as a self-healing mechanism."},
                {"text": "Dogs bark to communicate with humans and other dogs."},
            ],
            "top_n": 1,
            "model": model_name,
        })
        assert response.status_code == 200
        out = response.json()
        assert len(out["results"]) == 1

def test_rerank_return_documents():
    with TestClient(app) as client:
        model_name = RERANK_MODEL_NAMES[0]
        response = client.post("/v1/rerank", json={
            "query": "Why do cats purr?",
            "documents": [
                {"text": "Cats often purr when they feel relaxed and happy."},
                {"text": "Purring can also serve as a self-healing mechanism."},
                {"text": "Dogs bark to communicate with humans and other dogs."},
            ],
            "return_documents": True,
            "model": model_name,
        })
        assert response.status_code == 200
        out = response.json()
        for result in out["results"]:
            assert result.get("document") is not None
            assert "text" in result["document"]