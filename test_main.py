from fastapi.testclient import TestClient
from main import app, EMBED_MODEL_NAME, RERANK_MODEL_NAME


def test_read_healthz():
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200


def test_embedding_str():
    with TestClient(app) as client:
        embedding_request = {
            "input": "Cats often purr when they feel relaxed and happy.",
            "model": EMBED_MODEL_NAME,
        }
        response = client.post("/v1/embeddings", json=embedding_request)
        assert response.status_code == 200
        embedding_response = response.json()
        assert isinstance(embedding_response["data"], list)
        assert isinstance(embedding_response["data"][0]["embedding"], list)
        assert isinstance(embedding_response["data"][0]["embedding"][0], float)


def test_embedding_list_str():
    with TestClient(app) as client:
        embedding_request = {
            "input": [
                "Cats often purr when they feel relaxed and happy.",
                "Purring can also serve as a self-healing mechanism.",
            ],
            "model": EMBED_MODEL_NAME,
        }
        response = client.post("/v1/embeddings", json=embedding_request)
        assert response.status_code == 200
        embedding_response = response.json()
        assert isinstance(embedding_response["data"], list)
        assert isinstance(embedding_response["data"][0]["embedding"], list)
        assert isinstance(embedding_response["data"][0]["embedding"][0], float)

        assert isinstance(embedding_response["data"], list)
        assert isinstance(embedding_response["data"][1]["embedding"], list)
        assert isinstance(embedding_response["data"][1]["embedding"][0], float)

        embedding_1 = embedding_response["data"][0]["embedding"]
        embedding_2 = embedding_response["data"][1]["embedding"]
        assert embedding_1 != embedding_2


def test_rerank_basic():
    """Happy-path: full list returned and sorted by relevance_score."""
    with TestClient(app) as client:
        rerank_request = {
            "query": "Why do cats purr?",
            "documents": [
                {"text": "Cats often purr when they feel relaxed and happy."},
                {"text": "Purring can also serve as a self-healing mechanism."},
                {"text": "Dogs bark to communicate with humans and other dogs."},
            ],
            "model": RERANK_MODEL_NAME,
        }
        response = client.post("/v1/rerank", json=rerank_request)
        assert response.status_code == 200
        rerank_response = response.json()

        # Basic structure checks
        assert isinstance(rerank_response["results"], list)
        assert {result["index"] for result in rerank_response["results"]} == {0, 1, 2}

        # Ensure scores are in descending order
        scores = [result["relevance_score"] for result in rerank_response["results"]]
        assert scores == sorted(scores, reverse=True)


def test_rerank_top_n():
    """`top_n` parameter should limit number of returned results."""
    with TestClient(app) as client:
        rerank_request = {
            "query": "Why do cats purr?",
            "documents": [
                {"text": "Cats often purr when they feel relaxed and happy."},
                {"text": "Purring can also serve as a self-healing mechanism."},
                {"text": "Dogs bark to communicate with humans and other dogs."},
            ],
            "top_n": 1,
            "model": RERANK_MODEL_NAME,
        }
        response = client.post("/v1/rerank", json=rerank_request)
        assert response.status_code == 200
        rerank_response = response.json()
        assert len(rerank_response["results"]) == 1


def test_rerank_return_documents():
    """When `return_documents` is True, each result should include the original document."""
    with TestClient(app) as client:
        rerank_request = {
            "query": "Why do cats purr?",
            "documents": [
                {"text": "Cats often purr when they feel relaxed and happy."},
                {"text": "Purring can also serve as a self-healing mechanism."},
                {"text": "Dogs bark to communicate with humans and other dogs."},
            ],
            "return_documents": True,
            "model": RERANK_MODEL_NAME,
        }
        response = client.post("/v1/rerank", json=rerank_request)
        assert response.status_code == 200
        rerank_response = response.json()

        assert len(rerank_response["results"]) > 0
        # Verify that every result contains its corresponding document
        for result in rerank_response["results"]:
            assert result.get("document") is not None
            assert "text" in result["document"]