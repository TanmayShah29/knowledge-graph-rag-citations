"""LLM-based answer synthesis with inline span-level citations."""

from ..state import Citation, GraphTriple, Answer
from ..llm import invoke_llm

SYNTHESIS_PROMPT = [
    (
        "system",
        "You are a knowledge-graph-powered answer engine. Given a user question, "
        "a set of relevant graph triples, and source text spans, compose a factual "
        "answer. EVERY factual sentence MUST end with an inline citation marker "
        "[N] where N refers to the citation number. Citations must map to the "
        "provided source spans, not just source documents. If the evidence is "
        "insufficient, say so explicitly.\n\n"
        "Citation format: [N] where N is the citation index.\n"
        "Available citations:\n{citations_block}\n\n"
        "Return the answer text only. Each factual claim must have a citation.",
    ),
    (
        "user",
        "Question: {question}\n\n"
        "Relevant triples:\n{triples_block}\n\n"
        "Compose a cited answer.",
    ),
]


def _build_citations_block(citations: list[Citation]) -> str:
    lines = []
    for c in citations:
        lines.append(
            f"[{c.marker}] \"{c.span_text}\" — from {c.source_doc} "
            f"(chars {c.char_start}-{c.char_end})"
        )
    return "\n".join(lines)


def _build_triples_block(triples: list[GraphTriple]) -> str:
    lines = []
    for t in triples:
        lines.append(f"({t.subject}) --[{t.predicate}]--> ({t.obj})")
    return "\n".join(lines)


def synthesize_answer(
    question: str,
    subgraph: list[GraphTriple],
    citations: list[Citation],
) -> Answer:
    """Synthesize a cited answer from the collected subgraph and citations."""
    citations_block = _build_citations_block(citations)
    triples_block = _build_triples_block(subgraph)

    prompt = [
        (role, msg.format(
            question=question,
            citations_block=citations_block,
            triples_block=triples_block,
        ))
        for role, msg in SYNTHESIS_PROMPT
    ]
    answer_text = invoke_llm(prompt, temperature=0.0)
    return Answer(text=answer_text, citations=citations, subgraph=subgraph)
