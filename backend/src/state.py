"""Plain dataclasses for domain objects."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Triple:
    subject: str
    predicate: str
    obj: str
    source_doc: str = ""
    char_start: int = 0
    char_end: int = 0
    chunk_id: str = ""


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_doc: str
    char_start: int = 0
    char_end: int = 0
    embedding: list[float] = field(default_factory=list)


@dataclass
class VectorHit:
    chunk_id: str
    text: str
    source_doc: str
    char_start: int = 0
    char_end: int = 0
    score: float = 0.0


@dataclass
class GraphTriple:
    subject: str
    predicate: str
    obj: str
    source_doc: str = ""
    char_start: int = 0
    char_end: int = 0
    chunk_id: str = ""


@dataclass
class Citation:
    marker: int
    source_doc: str
    char_start: int
    char_end: int
    span_text: str
    triple: Optional[GraphTriple] = None


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    subgraph: list[GraphTriple] = field(default_factory=list)
    verified: bool = False
    verification_notes: list[str] = field(default_factory=list)


@dataclass
class GraphNode:
    name: str
    label: str = "Entity"


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
