"""Neo4j graph store — constraints, triple upserts, and traversal queries."""

from __future__ import annotations
from typing import List
from neo4j import GraphDatabase
from ..config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, MAX_HOPS
from ..state import Triple, GraphTriple

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def close():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def ensure_constraints():
    with _get_driver().session(database=NEO4J_DATABASE) as session:
        session.run(
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        )


def upsert_triple(triple: Triple) -> None:
    query = """
    MERGE (s:Entity {name: $subject})
    MERGE (o:Entity {name: $obj})
    MERGE (s)-[r:RELATES_TO {predicate: $predicate}]->(o)
    SET r.source_doc = $source_doc,
        r.char_start = $char_start,
        r.char_end = $char_end,
        r.chunk_id = $chunk_id
    """
    with _get_driver().session(database=NEO4J_DATABASE) as session:
        session.run(
            query,
            subject=triple.subject,
            obj=triple.obj,
            predicate=triple.predicate,
            source_doc=triple.source_doc,
            char_start=triple.char_start,
            char_end=triple.char_end,
            chunk_id=triple.chunk_id,
        )


def upsert_triples(triples: List[Triple]) -> None:
    for t in triples:
        upsert_triple(t)


def find_entity_nodes(names: List[str]) -> List[dict]:
    """Return matching Entity nodes for a list of candidate names."""
    query = """
    MATCH (e:Entity)
    WHERE e.name IN $names
    RETURN e.name AS name
    """
    with _get_driver().session(database=NEO4J_DATABASE) as session:
        result = session.run(query, names=names)
        return [dict(r) for r in result]


def traverse_from_entities(
    entity_names: List[str], max_hops: int = MAX_HOPS
) -> List[GraphTriple]:
    """Run a multi-hop BFS from seed entities and return the collected subgraph."""
    query = """
    UNWIND $seeds AS seed_name
    MATCH (start:Entity {name: seed_name})
    CALL apoc.path.subgraphAll(start, {
        maxLevel: $max_hops,
        relationshipFilter: 'RELATES_TO'
    }) YIELD relationships
    UNWIND relationships AS rel
    MATCH (a)-[rel]->(b)
    RETURN DISTINCT a.name AS subject,
           rel.predicate AS predicate,
           b.name AS obj,
           rel.source_doc AS source_doc,
           rel.char_start AS char_start,
           rel.char_end AS char_end,
           rel.chunk_id AS chunk_id
    """
    with _get_driver().session(database=NEO4J_DATABASE) as session:
        result = session.run(query, seeds=entity_names, max_hops=max_hops)
        return [
            GraphTriple(
                subject=r["subject"],
                predicate=r["predicate"],
                obj=r["obj"],
                source_doc=r.get("source_doc", ""),
                char_start=r.get("char_start", 0),
                char_end=r.get("char_end", 0),
                chunk_id=r.get("chunk_id", ""),
            )
            for r in result
        ]


def traverse_programmatic(
    entity_names: List[str], max_hops: int = MAX_HOPS
) -> List[GraphTriple]:
    """BFS traversal without APOC — works on vanilla Neo4j."""
    visited_rels = set()
    results: List[GraphTriple] = []
    current_seeds = list(entity_names)

    with _get_driver().session(database=NEO4J_DATABASE) as session:
        for hop in range(max_hops):
            if not current_seeds:
                break
            query = """
            UNWIND $seeds AS seed_name
            MATCH (a:Entity {name: seed_name})-[r:RELATES_TO]->(b:Entity)
            RETURN a.name AS subject, r.predicate AS predicate,
                   b.name AS obj, r.source_doc AS source_doc,
                   r.char_start AS char_start, r.char_end AS char_end,
                   r.chunk_id AS chunk_id
            """
            result = session.run(query, seeds=current_seeds)
            next_seeds = []
            for r in result:
                key = (r["subject"], r["predicate"], r["obj"])
                if key not in visited_rels:
                    visited_rels.add(key)
                    results.append(
                        GraphTriple(
                            subject=r["subject"],
                            predicate=r["predicate"],
                            obj=r["obj"],
                            source_doc=r.get("source_doc", ""),
                            char_start=r.get("char_start", 0),
                            char_end=r.get("char_end", 0),
                            chunk_id=r.get("chunk_id", ""),
                        )
                    )
                    next_seeds.append(r["obj"])
            current_seeds = list(set(next_seeds) - set(entity_names))

    return results


def traverse_llm_cypher(
    entity_names: List[str], max_hops: int = MAX_HOPS
) -> List[GraphTriple]:
    """Use LLM to generate a Cypher query for traversal, then execute it."""
    from ..llm import invoke_llm

    seed_list = ", ".join(f'"{n}"' for n in entity_names)
    prompt = [
        (
            "system",
            "You are a Cypher query generator. Given seed entity names and a max hop count, "
            "write a single Cypher query that traverses RELATES_TO relationships from those "
            "seeds up to max_hops levels deep. Return columns: subject, predicate, obj, "
            "source_doc, char_start, char_end, chunk_id. Use UNWIND for multiple seeds. "
            "Do not use APOC. Output ONLY the Cypher query, nothing else.",
        ),
        (
            "user",
            f"Seeds: [{seed_list}]\nMax hops: {max_hops}\n"
            "Schema: (Entity)-[:RELATES_TO {predicate, source_doc, char_start, char_end, chunk_id}]->(Entity)",
        ),
    ]
    cypher = invoke_llm(prompt, temperature=0.0).strip().strip("```cypher").strip("```")

    with _get_driver().session(database=NEO4J_DATABASE) as session:
        result = session.run(cypher)
        return [
            GraphTriple(
                subject=r["subject"],
                predicate=r["predicate"],
                obj=r["obj"],
                source_doc=r.get("source_doc", ""),
                char_start=r.get("char_start", 0),
                char_end=r.get("char_end", 0),
                chunk_id=r.get("chunk_id", ""),
            )
            for r in result
        ]


def get_triple_by_chunk(chunk_id: str) -> List[GraphTriple]:
    """Get all triples sourced from a specific chunk."""
    query = """
    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
    WHERE r.chunk_id = $chunk_id
    RETURN a.name AS subject, r.predicate AS predicate, b.name AS obj,
           r.source_doc AS source_doc, r.char_start AS char_start,
           r.char_end AS char_end, r.chunk_id AS chunk_id
    """
    with _get_driver().session(database=NEO4J_DATABASE) as session:
        result = session.run(query, chunk_id=chunk_id)
        return [
            GraphTriple(
                subject=r["subject"],
                predicate=r["predicate"],
                obj=r["obj"],
                source_doc=r.get("source_doc", ""),
                char_start=r.get("char_start", 0),
                char_end=r.get("char_end", 0),
                chunk_id=r.get("chunk_id", ""),
            )
            for r in result
        ]
