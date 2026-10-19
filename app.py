from fastapi import FastAPI
from pydantic import BaseModel, Field
import re

app = FastAPI(title="Advanced Hybrid RAG with Reranking", version="0.1.0")
DOCUMENTS = [
 {"id":"doc-1","title":"RAG overview","text":"Retrieval augmented generation retrieves relevant documents and uses them to ground generated answers."},
 {"id":"doc-2","title":"Evaluation guide","text":"Evaluate retrieval with recall, ranking quality, and citation correctness."},
 {"id":"doc-3","title":"Safety guide","text":"Use permission checks, prompt injection defenses, and human review for risky actions."},
]

class Query(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=3, ge=1, le=10)

def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def retrieve(query: str, top_k: int = 3) -> dict:
    q = tokens(query)
    lexical = sorted(DOCUMENTS, key=lambda d: len(q & tokens(d["title"] + " " + d["text"])), reverse=True)
    semantic_proxy = sorted(DOCUMENTS, key=lambda d: len(q & tokens(d["text"])) / max(1, len(q | tokens(d["text"]))), reverse=True)
    ranks = {}
    for ranked in (lexical, semantic_proxy):
        for rank, doc in enumerate(ranked, 1):
            ranks[doc["id"]] = ranks.get(doc["id"], 0) + 1 / (60 + rank)
    by_id = {d["id"]: d for d in DOCUMENTS}
    ordered = sorted(ranks, key=lambda key: (ranks[key], len(q & tokens(by_id[key]["title"] + " " + by_id[key]["text"]))), reverse=True)
    return {"query": query, "results":[{**by_id[key], "rrf_score": round(ranks[key], 5)} for key in ordered[:top_k]],
            "method":"lexical overlap + Jaccard proxy + reciprocal-rank fusion"}

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/documents")
def documents(): return {"documents": DOCUMENTS}

@app.post("/retrieve")
def retrieve_endpoint(query: Query): return retrieve(query.query, query.top_k)
