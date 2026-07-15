"""Query-time retrieval: embed question → vector search → graph traversal → synthesize."""

from __future__ import annotations
from .config import TOP_K_CHUNKS, MAX_HOPS
from .state import Citation, GraphTriple, Answer
from .embeddings import embed_text
from .vector import vector_search
from .graph import traverse_programmatic, traverse_llm_cypher, get_triple_by_chunk
from .agents.synthesizer import synthesize_answer
from .agents.verifier import verify_answer


def _build_citations_from_hits(hits, chunks_map) -> list[Citation]:
    """Build Citation objects from vector search hits."""
    citations = []
    seen = set()
    for i, hit in enumerate(hits):
        key = (hit.source_doc, hit.char_start, hit.char_end)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(
            marker=len(citations) + 1,
            source_doc=hit.source_doc,
            char_start=hit.char_start,
            char_end=hit.char_end,
            span_text=hit.text[:500],
        ))
    return citations


def query(
    question: str,
    max_hops: int = MAX_HOPS,
    traversal_mode: str = "programmatic",
    verify: bool = True,
) -> Answer:
    """End-to-end query: vector search → graph traversal → synthesis → verification."""

    # 1. Embed the question
    q_embedding = embed_text(question)

    # 2. Vector search for entry points
    hits = vector_search(q_embedding, top_k=TOP_K_CHUNKS)
    if not hits:
        return Answer(text="No relevant documents found.", verified=False)

    # 3. Extract entity names from matched chunks via their triples
    entity_names = set()
    entry_chunks = set()
    for hit in hits:
        entry_chunks.add(hit.chunk_id)
        triples = get_triple_by_chunk(hit.chunk_id)
        for t in triples:
            entity_names.add(t.subject)
            entity_names.add(t.obj)

    if not entity_names:
        # Fallback: try to extract entity names from chunk text via simple NER
        for hit in hits:
            words = hit.text.split()
            for w in words:
                if w[0].isupper() and len(w) > 2:
                    entity_names.add(w)

    entity_names = list(entity_names)[:20]  # cap to avoid huge traversals

    # 4. Graph traversal
    if traversal_mode == "llm_cypher":
        subgraph = traverse_llm_cypher(entity_names, max_hops=max_hops)
    else:
        subgraph = traverse_programmatic(entity_names, max_hops=max_hops)

    # If traversal returned nothing, fall back to just the entry chunk triples
    if not subgraph:
        for hit in hits:
            subgraph.extend(get_triple_by_chunk(hit.chunk_id))

    # 5. Build citations from both vector hits and traversal subgraph
    citations = _build_citations_from_hits(hits, {})

    # Add citations from subgraph triples not yet covered
    seen_spans = {(c.source_doc, c.char_start, c.char_end) for c in citations}
    for t in subgraph:
        key = (t.source_doc, t.char_start, t.char_end)
        if key not in seen_spans and t.source_doc:
            seen_spans.add(key)
            citations.append(Citation(
                marker=len(citations) + 1,
                source_doc=t.source_doc,
                char_start=t.char_start,
                char_end=t.char_end,
                span_text=f"({t.subject}) --[{t.predicate}]--> ({t.obj})",
                triple=t,
            ))

    # 6. Synthesize answer
    answer = synthesize_answer(question, subgraph, citations)

    # 7. Verify
    if verify:
        answer = verify_answer(answer)

    return answer


def query_single_hop(question: str, verify: bool = True) -> Answer:
    """Single-hop query (no graph traversal)."""
    return query(question, max_hops=0, verify=verify)


def query_multi_hop(
    question: str,
    max_hops: int = MAX_HOPS,
    traversal_mode: str = "programmatic",
    verify: bool = True,
) -> Answer:
    """Multi-hop query with configurable traversal."""
    return query(question, max_hops=max_hops, traversal_mode=traversal_mode, verify=verify)
