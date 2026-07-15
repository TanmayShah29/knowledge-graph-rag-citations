"""FastAPI application — GraphCite API."""

from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .ingest import ingest_text, ingest_file, ingest_directory
from .retrieval import query, query_single_hop, query_multi_hop
from .graph import ensure_constraints, traverse_programmatic
from .vector import ensure_table
from .config import MAX_HOPS

app = FastAPI(title="GraphCite", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    max_hops: int = MAX_HOPS
    traversal_mode: str = "programmatic"  # or "llm_cypher"
    verify: bool = True


class IngestTextRequest(BaseModel):
    text: str
    source_doc: str = "manual_input"


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    subgraph: list[dict]
    verified: bool
    verification_notes: list[str]


@app.on_event("startup")
def startup():
    ensure_constraints()
    ensure_table()


@app.post("/ingest/text")
def ingest_text_endpoint(req: IngestTextRequest):
    result = ingest_text(req.text, req.source_doc)
    return {"status": "ok", **result}


@app.post("/ingest/file")
async def ingest_file_endpoint(file: UploadFile = File(...)):
    import tempfile, os
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = ingest_file(tmp_path)
    finally:
        os.unlink(tmp_path)
    return {"status": "ok", **result}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    answer = query(
        req.question,
        max_hops=req.max_hops,
        traversal_mode=req.traversal_mode,
        verify=req.verify,
    )
    return QueryResponse(
        answer=answer.text,
        citations=[{
            "marker": c.marker,
            "source_doc": c.source_doc,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "span_text": c.span_text,
        } for c in answer.citations],
        subgraph=[{
            "subject": t.subject,
            "predicate": t.predicate,
            "object": t.obj,
            "source_doc": t.source_doc,
        } for t in answer.subgraph],
        verified=answer.verified,
        verification_notes=answer.verification_notes,
    )


@app.get("/graph/traversal")
def graph_traversal_endpoint(entity: str, hops: int = 2):
    subgraph = traverse_programmatic([entity], max_hops=hops)
    return {
        "nodes": list({t.subject for t in subgraph} | {t.obj for t in subgraph}),
        "edges": [{"source": t.subject, "target": t.obj, "relation": t.predicate} for t in subgraph],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
