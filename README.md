# GraphCite — Knowledge Graph RAG with Verifiable Span-Level Citations

GraphCite is a multi-hop knowledge graph RAG system that answers complex questions over a document corpus with **verifiable, span-level citations**. Every factual claim in the answer is linked to the exact source span it came from, and a verification pass checks that each cited span actually supports the claim.

## Architecture

```
User Question
  │
  ▼
┌─────────────┐    ┌────────────┐
│  pgvector   │───▶│  Neo4j     │
│  (semantic  │    │  (graph    │
│   search)   │    │  traversal)│
└─────────────┘    └────────────┘
       │                │
       ▼                ▼
┌──────────────────────────────┐
│     LLM Answer Synthesis     │
│  (every claim → [N] marker)  │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│     Verification Pass        │
│  (NLI: is claim entailed     │
│   by cited span?)            │
└──────────────────────────────┘
       │
       ▼
    Answer + Citations + Graph Viz
```

### Components

| Service | Technology | Role |
|---------|-----------|------|
| **API** | Python, FastAPI, LangChain | Ingestion, retrieval, synthesis, verification |
| **Graph Store** | Neo4j 5 | Entity/relation triples with span provenance |
| **Vector Store** | Supabase pgvector | Semantic chunk embeddings for entry point retrieval |
| **LLM** | Groq + Llama 3.3 | Triple extraction, Cypher generation, answer synthesis, verification |
| **Frontend** | Next.js 16, TypeScript, Tailwind CSS | Chat UI, graph visualization (ForceGraph2D), citation click-through |

## Pipeline

### 1. Ingestion
- Accept PDFs or plain text documents
- Sliding-window chunking (configurable size/overlap)
- LLM extracts `(entity, relation, entity)` triples + source span (`doc_id`, `char_start`, `char_end`)
- Triples → Neo4j nodes/edges
- Chunks + embeddings → pgvector

### 2. Query
1. **Entry point retrieval**: embed question → top-k chunks via pgvector cosine similarity
2. **Multi-hop traversal**: from entry entities, run graph BFS (2-3 hops, configurable)
   - *Programmatic Cypher*: deterministic BFS via repeated queries
   - *LLM-Generated Cypher*: LLM writes the Cypher query for flexible traversal
3. **Subgraph collection**: gather relevant triples + source spans
4. **Answer synthesis**: LLM composes answer with inline `[N]` citations
5. **Verification**: second LLM pass checks each claim is entailed by its cited span

### 3. Frontend Features
- **Chat interface** with inline citation buttons
- **Click [N]** → highlights exact source span in side panel
- **Graph visualization** shows the traversal path used to answer
- **Verification status** badge (all verified / some unsupported)

## Setup

### Prerequisites
- Python 3.12+
- Node.js 22+
- Neo4j 5 (local Docker or AuraDB)
- PostgreSQL with pgvector extension (local or Supabase)
- Groq API key

### Backend

```bash
cd backend
cp .env.example .env
# Edit .env with your credentials
pip install -r requirements.txt

# Start services
docker run --name neo4j -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
  neo4j:5-enterprise

docker run --name pgvector -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  pgvector/pgvector:pg17

# Start API
uvicorn src.api:app --reload
```

### Frontend

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

### Ingest the demo corpus

```bash
curl -X POST http://localhost:8000/ingest/dir \
  -H "Content-Type: application/json" \
  -d '{"dir_path": "samples"}'
```

Or via Python:
```python
from src.ingest import ingest_directory
ingest_directory("samples")
```

## Demo Corpus

The demo corpus contains 6 Wikipedia-style articles on a connected topic: **technology companies and their founders, products, and leadership history**.

| Document | Entities |
|----------|----------|
| Apple Inc. | Steve Jobs, Steve Wozniak, Ronald Wayne, Tim Cook, Atari |
| Microsoft | Bill Gates, Paul Allen, Satya Nadella, Steve Ballmer |
| Google | Larry Page, Sergey Brin, Sundar Pichai, Alphabet Inc. |
| Amazon | Jeff Bezos, Andy Jassy, AWS, Whole Foods |
| Meta Platforms | Mark Zuckerberg, Sheryl Sandberg, Andrew Bosworth |
| Tesla | Elon Musk, Martin Eberhard, Marc Tarpenning, SpaceX |

This corpus enables meaningful multi-hop questions like:
- "What company did the founder of X previously work at, and who now runs that company?"
- "Which CEO left a hedge fund to start their company, and what grocery chain did they later acquire?"

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest/text` | Ingest raw text |
| POST | `/ingest/file` | Upload a file (PDF/txt) |
| POST | `/query` | Ask a question with multi-hop traversal |
| GET | `/graph/traversal` | Get graph subgraph for an entity |
| GET | `/health` | Health check |

### POST /query

```json
{
  "question": "What company did the founder of Tesla previously co-found?",
  "max_hops": 2,
  "traversal_mode": "programmatic",
  "verify": true
}
```

Response:
```json
{
  "answer": "Elon Musk, the founder and CEO of Tesla, previously co-founded PayPal [1][2]. PayPal was acquired by eBay in 2002 for $1.5 billion [2].",
  "citations": [
    {
      "marker": 1,
      "source_doc": "tesla.txt",
      "char_start": 234,
      "char_end": 300,
      "span_text": "Elon Musk became CEO of Tesla in 2008... Before Tesla, Elon Musk co-founded PayPal in 1999..."
    }
  ],
  "verified": true,
  "verification_notes": ["All citations verified"]
}
```

## Eval Metrics

Run evaluation on the demo corpus:

```bash
python eval/run_eval.py --ingest backend/samples
```

Expected metrics format:

```json
{
  "num_questions": 14,
  "avg_latency_s": 3.2,
  "hop_accuracy": 0.857,
  "citation_precision": 0.925,
  "total_citations": 40,
  "supported_citations": 37
}
```

## Docker

```bash
docker compose up --build
```

This starts:
- `api` on `:8000`
- `neo4j` on `:7687`
- `db` (pgvector) on `:5432`
- `web` on `:3000`

## Deployment

- **API**: Render, Fly.io, or any cloud VM — single `uvicorn src.api:app` process
- **Frontend**: Vercel — `cd frontend && vercel --prod`
- **Neo4j**: AuraDB free tier (cloud) or Docker
- **pgvector**: Supabase free tier or cloud PostgreSQL

## License

MIT

# knowledge-graph-rag-citations
