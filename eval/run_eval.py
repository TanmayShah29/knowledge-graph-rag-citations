"""Evaluation script — hop accuracy, citation precision, latency."""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import query
from src.ingest import ingest_directory


TEST_CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"


def load_test_cases() -> list[dict]:
    if TEST_CASES_PATH.exists():
        return json.loads(TEST_CASES_PATH.read_text())
    return []


def run_eval(test_cases: list[dict] | None = None) -> dict:
    """Run evaluation over test cases and compute metrics."""
    if test_cases is None:
        test_cases = load_test_cases()

    results = []
    total_latency = 0.0
    correct_hops = 0
    supported_citations = 0
    total_citations = 0

    for tc in test_cases:
        question = tc["question"]
        expected_path = tc.get("expected_path", [])
        expected_answer_keywords = tc.get("expected_keywords", [])

        start = time.time()
        answer = query(question, verify=True)
        latency = time.time() - start
        total_latency += latency

        # Hop accuracy: did the subgraph contain the expected entities?
        subgraph_entities = set()
        for t in answer.subgraph:
            subgraph_entities.add(t.subject.lower())
            subgraph_entities.add(t.obj.lower())

        hop_correct = True
        if expected_path:
            for entity in expected_path:
                if entity.lower() not in subgraph_entities:
                    hop_correct = False
                    break
        if hop_correct:
            correct_hops += 1

        # Citation precision
        total_citations += len(answer.citations)
        if answer.verified:
            supported_citations += len(answer.citations)

        results.append({
            "question": question,
            "answer": answer.text,
            "verified": answer.verified,
            "num_citations": len(answer.citations),
            "num_triples": len(answer.subgraph),
            "latency_s": round(latency, 3),
            "hop_correct": hop_correct,
        })

    n = len(test_cases) or 1
    metrics = {
        "num_questions": len(test_cases),
        "avg_latency_s": round(total_latency / n, 3),
        "hop_accuracy": round(correct_hops / n, 3),
        "citation_precision": round(supported_citations / max(total_citations, 1), 3),
        "total_citations": total_citations,
        "supported_citations": supported_citations,
        "results": results,
    }
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GraphCite evaluation")
    parser.add_argument("--ingest", type=str, help="Directory of documents to ingest first")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON path")
    args = parser.parse_args()

    if args.ingest:
        print(f"Ingesting from {args.ingest}...")
        results = ingest_directory(args.ingest)
        for r in results:
            print(f"  {r['source_doc']}: {r['num_chunks']} chunks, {r['num_triples']} triples")

    print("Running evaluation...")
    metrics = run_eval()
    print(json.dumps(metrics, indent=2))

    out_path = Path(args.output)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nResults written to {out_path}")
