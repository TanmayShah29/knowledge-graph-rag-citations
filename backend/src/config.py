"""Centralised configuration — all knobs in one place."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Neo4j ────────────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ── Supabase pgvector ────────────────────────────────────────────────────────
PGVECTOR_URL = os.getenv(
    "PGVECTOR_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres",
)
PGVECTOR_URL_SYNC = os.getenv(
    "PGVECTOR_URL_SYNC",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "document_chunks")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

# ── LLM (Groq + Llama 3.3) ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "groq")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2")

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "5"))
TOP_K_ENTITIES = int(os.getenv("TOP_K_ENTITIES", "10"))
MAX_HOPS = int(os.getenv("MAX_HOPS", "2"))

# ── Ingestion ────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
