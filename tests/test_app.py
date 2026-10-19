from app import retrieve

def test_retrieval_returns_provenance_and_scores():
    result = retrieve("prompt injection safety", 2)
    assert result["results"][0]["id"] == "doc-3"
    assert "rrf_score" in result["results"][0]
    assert result["method"].startswith("lexical overlap")

def test_top_k_is_respected():
    assert len(retrieve("RAG retrieval", 1)["results"]) == 1
