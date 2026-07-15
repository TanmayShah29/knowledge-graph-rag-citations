"""Supabase pgvector store — chunk storage and semantic search."""

from __future__ import annotations
from sqlalchemy import (
    Column, String, Text, Integer, Float, create_engine, text
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from ..config import (
    PGVECTOR_URL, PGVECTOR_URL_SYNC, PGVECTOR_TABLE,
    EMBEDDING_DIMENSIONS, TOP_K_CHUNKS,
)
from ..state import Chunk, VectorHit

Base = declarative_base()

_engine = None
_async_engine = None
_SessionLocal = None
_AsyncSessionLocal = None


class ChunkRow(Base):
    __tablename__ = PGVECTOR_TABLE

    chunk_id = Column(String, primary_key=True)
    text = Column(Text, nullable=False)
    source_doc = Column(String, nullable=False)
    char_start = Column(Integer, default=0)
    char_end = Column(Integer, default=0)
    embedding = Column(Vector(EMBEDDING_DIMENSIONS))


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(PGVECTOR_URL_SYNC)
    return _engine


def _get_async_engine():
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(PGVECTOR_URL)
    return _async_engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_get_engine())
    return _SessionLocal


def _get_async_session_factory():
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        _AsyncSessionLocal = sessionmaker(
            class_=AsyncSession, bind=_get_async_engine(), expire_on_commit=False
        )
    return _AsyncSessionLocal


def ensure_table():
    engine = _get_engine()

    # Enable pgvector extension first (must exist before VECTOR type is used)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(engine)


def upsert_chunk(chunk: Chunk) -> None:
    Session = _get_session_factory()
    with Session() as session:
        existing = session.get(ChunkRow, chunk.chunk_id)
        if existing:
            existing.text = chunk.text
            existing.source_doc = chunk.source_doc
            existing.char_start = chunk.char_start
            existing.char_end = chunk.char_end
            existing.embedding = chunk.embedding
        else:
            row = ChunkRow(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_doc=chunk.source_doc,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                embedding=chunk.embedding,
            )
            session.add(row)
        session.commit()


def upsert_chunks(chunks: list[Chunk]) -> None:
    for c in chunks:
        upsert_chunk(c)


def vector_search(query_embedding: list[float], top_k: int = TOP_K_CHUNKS) -> list[VectorHit]:
    """Cosine similarity search over stored chunk embeddings."""
    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text(f"""
                SELECT chunk_id, text, source_doc, char_start, char_end,
                       1 - (embedding <=> :query_vec::vector) AS score
                FROM {PGVECTOR_TABLE}
                ORDER BY embedding <=> :query_vec::vector
                LIMIT :k
            """),
            {"query_vec": str(query_embedding), "k": top_k},
        )
        rows = result.fetchall()
    return [
        VectorHit(
            chunk_id=r[0],
            text=r[1],
            source_doc=r[2],
            char_start=r[3],
            char_end=r[4],
            score=float(r[5]),
        )
        for r in rows
    ]
