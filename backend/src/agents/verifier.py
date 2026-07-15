"""Post-generation verification: check each cited sentence is entailed by its span."""

from ..state import Answer, Citation
from ..llm import invoke_llm

VERIFICATION_PROMPT = [
    (
        "system",
        "You are a citation verification system. For each numbered claim in the "
        "answer, check whether the claim is actually supported (entailed) by the "
        "text of its cited span. Return a JSON array of objects with keys: "
        "marker (int), supported (bool), note (str). Output ONLY valid JSON.",
    ),
    (
        "user",
        "Answer:\n{answer}\n\n"
        "Citations:\n{citations_block}",
    ),
]


def _build_citations_block(citations: list[Citation]) -> str:
    lines = []
    for c in citations:
        lines.append(f"[{c.marker}] Span text: \"{c.span_text}\"")
    return "\n".join(lines)


def verify_answer(answer: Answer) -> Answer:
    """Run NLI-style verification on each citation in the answer."""
    citations_block = _build_citations_block(answer.citations)
    prompt = [
        (role, msg.format(answer=answer.text, citations_block=citations_block))
        for role, msg in VERIFICATION_PROMPT
    ]
    response = invoke_llm(prompt, temperature=0.0)

    import json
    try:
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        checks = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        answer.verified = False
        answer.verification_notes = ["Verification parse failed"]
        return answer

    unsupported = []
    for check in checks:
        if isinstance(check, dict) and not check.get("supported", True):
            unsupported.append(f"[{check.get('marker', '?')}] {check.get('note', 'unsupported')}")

    answer.verified = len(unsupported) == 0
    answer.verification_notes = unsupported if unsupported else ["All citations verified"]
    return answer
