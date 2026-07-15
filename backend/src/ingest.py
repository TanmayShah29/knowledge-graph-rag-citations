"""Ingestion pipeline: documents → chunks → triples → Neo4j + pgvector."""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import List

from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .state import Chunk, Triple
from .embeddings import embed_texts
from .graph import ensure_constraints, upsert_triples
from .vector import ensure_table, upsert_chunks
from .agents.extractor import extract_all


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Sliding-window chunking with overlap."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end]
        chunks.append({
            "text": chunk_text,
            "char_start": start,
            "char_end": end,
            "index": idx,
        })
        start += size - overlap
        idx += 1
    return chunks


def _chunk_id(source_doc: str, char_start: int, text: str) -> str:
    return hashlib.sha256(f"{source_doc}:{char_start}:{text[:80]}".encode()).hexdigest()[:16]


def ingest_text(text: str, source_doc: str) -> dict:
    """Full ingestion pipeline for a single text document."""
    # 1. Ensure stores are ready
    ensure_constraints()
    ensure_table()

    # 2. Chunk
    raw_chunks = _chunk_text(text)
    for c in raw_chunks:
        c["source_doc"] = source_doc

    # 3. Embed chunks
    texts = [c["text"] for c in raw_chunks]
    embeddings = embed_texts(texts)

    # 4. Build Chunk objects and upsert to pgvector
    chunks = []
    for c, emb in zip(raw_chunks, embeddings):
        cid = _chunk_id(source_doc, c["char_start"], c["text"])
        chunks.append(Chunk(
            chunk_id=cid,
            text=c["text"],
            source_doc=source_doc,
            char_start=c["char_start"],
            char_end=c["char_end"],
            embedding=emb,
        ))
    upsert_chunks(chunks)

    # 5. Extract triples via LLM
    triples = extract_all(raw_chunks)

    # 6. Write triples to Neo4j
    upsert_triples(triples)

    return {
        "source_doc": source_doc,
        "num_chunks": len(chunks),
        "num_triples": len(triples),
    }


def ingest_file(file_path: str) -> dict:
    """Ingest a single file (PDF or text)."""
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return _ingest_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        return ingest_text(text, source_doc=path.name)


def _ingest_pdf(path: Path) -> dict:
    """Extract text from PDF and ingest."""
    try:
        import pymupdf
        doc = pymupdf.open(str(path))
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
    except ImportError:
        raise ImportError("pip install pymupdf for PDF support")
    return ingest_text(full_text, source_doc=path.name)


def ingest_directory(dir_path: str) -> list[dict]:
    """Ingest all .txt and .pdf files in a directory."""
    results = []
    p = Path(dir_path)
    for f in sorted(p.glob("*")):
        if f.suffix.lower() in (".txt", ".pdf"):
            results.append(ingest_file(str(f)))
    return results
