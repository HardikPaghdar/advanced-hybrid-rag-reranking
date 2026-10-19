# Advanced Hybrid RAG with Reranking

An inspectable retrieval prototype that combines lexical matching, a Jaccard-overlap proxy, and a reciprocal-rank-fusion-style ranking signal.

## Features
- Small editable document corpus with IDs and titles
- Transparent lexical and overlap ranking signals
- Ranked results include source IDs, text, and scores
- Configurable top-k retrieval
- API docs, tests, and CI

## Start
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Try `POST /retrieve` in http://127.0.0.1:8000/docs.

## Important limitations
The overlap proxy is not dense-vector semantic search. This version has no embedding model, vector database, query rewriting, or LLM answer generation. Keeping each ranking signal visible makes it easier to add and evaluate more advanced retrieval techniques.

## Stack
Python, FastAPI, Pydantic, pytest. No paid APIs required.
