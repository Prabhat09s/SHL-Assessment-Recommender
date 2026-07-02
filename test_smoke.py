"""
Quick smoke test - checks the app boots, /health works, catalog loads
and retrieval returns sensible results, and /chat responds gracefully
even with no GROQ_API_KEY set (fallback path). Run with:

    python test_smoke.py

For a real end-to-end test of the conversational behavior, set
GROQ_API_KEY and use test_conversation.py instead.
"""
from fastapi.testclient import TestClient

from app.main import app, catalog

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("[OK] /health")


def test_catalog_loaded():
    assert catalog is not None
    assert len(catalog) > 0
    print(f"[OK] catalog loaded with {len(catalog)} items")


def test_retrieval_relevance():
    results = catalog.search("senior Java developer Spring SQL", top_k=5)
    names = [item["name"] for item, _score in results]
    print("[INFO] top matches for Java/Spring/SQL query:", names)
    assert any("Java" in n or "Spring" in n or "SQL" in n for n in names)
    print("[OK] retrieval returns topically relevant results")


def test_chat_endpoint_shape():
    r = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "We need a solution for senior leadership."}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert "reply" in body
    assert "recommendations" in body
    assert "end_of_conversation" in body
    assert isinstance(body["recommendations"], list)
    print("[OK] /chat returns schema-correct response:", body)


if __name__ == "__main__":
    test_health()
    test_catalog_loaded()
    test_retrieval_relevance()
    test_chat_endpoint_shape()
    print("\nAll smoke tests passed.")
