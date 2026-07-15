"""LLM-based triple extraction from document chunks."""

import json
import hashlib
from ..state import Triple
from ..llm import invoke_llm

EXTRACTION_PROMPT = [
    (
        "system",
        "You are a knowledge graph extraction engine. Given a text chunk, extract "
        "all factual triples in the form (subject, predicate, object). Each triple "
        "must capture a single atomic fact. Use clear, normalized entity names. "
        "Return a JSON array of objects with keys: subject, predicate, object. "
        "Output ONLY valid JSON, no markdown fences, no commentary.",
    ),
    (
        "user",
        "Text:\n{text}\n\n"
        "Extract all (subject, predicate, object) triples. Return JSON array only.",
    ),
]


def _make_chunk_id(source_doc: str, char_start: int, text: str) -> str:
    h = hashlib.sha256(f"{source_doc}:{char_start}:{text[:100]}".encode()).hexdigest()[:16]
    return h


def extract_triples(chunk_text: str, source_doc: str, char_start: int, char_end: int) -> list[Triple]:
    """Extract triples from a single text chunk using the LLM."""
    prompt = [(role, msg.format(text=chunk_text)) for role, msg in EXTRACTION_PROMPT]
    response = invoke_llm(prompt, temperature=0.0)

    # Parse JSON response
    try:
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        triples_data = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        return []

    chunk_id = _make_chunk_id(source_doc, char_start, chunk_text)
    triples = []
    for t in triples_data:
        if isinstance(t, dict) and all(
            k in t and isinstance(t[k], str) and t[k].strip()
            for k in ("subject", "predicate", "object")
        ):
            triples.append(
                Triple(
                    subject=t["subject"].strip(),
                    predicate=t["predicate"].strip(),
                    obj=t["object"].strip(),
                    source_doc=source_doc,
                    char_start=char_start,
                    char_end=char_end,
                    chunk_id=chunk_id,
                )
            )
    return triples


def extract_all(chunks: list[dict]) -> list[Triple]:
    """Extract triples from multiple chunks. Each chunk dict has keys:
    text, source_doc, char_start, char_end."""
    all_triples = []
    for chunk in chunks:
        triples = extract_triples(
            chunk["text"], chunk["source_doc"], chunk["char_start"], chunk["char_end"]
        )
        all_triples.extend(triples)
    return all_triples
