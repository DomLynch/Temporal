# temporal

> Graphiti's temporal knowledge graph. 53,000 lines stripped to 2,800.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License-Apache--2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-155%20passing-brightgreen.svg)](#tests)

Temporal is a knowledge graph that knows *when* things were true. It extracts entities and relations from text, tracks how facts change over time, and lets you query the state of the world at any point in history.

```python
from temporal import retain, search

# Learn a fact
await retain("Alice joined Acme Corp in March 2024", store=store, llm=llm, embedder=embedder)

# Later...
await retain("Alice left Acme Corp in January 2025", store=store, llm=llm, embedder=embedder)

# Query: what's true now?
results = await search("where does Alice work?", store=store, embedder=embedder)
# → returns the January 2025 relation (Alice LEFT Acme), old relation is invalidated
```

No Neo4j. No graph database. No Docker. Just SQLite.

---

## Size comparison

| Component | Graphiti | Temporal | Reduction |
|-----------|----------|----------|-----------|
| Neo4j driver | 2,785 LOC | — | 100% |
| FalkorDB driver | 2,883 LOC | — | 100% |
| Kuzu driver | 2,889 LOC | — | 100% |
| Neptune driver | 2,816 LOC | — | 100% |
| All graph DB drivers | 13,109 LOC | **430 LOC** (SQLite) | **97%** |
| LLM adapter + prompts | ~3,000 LOC | 417 LOC | 86% |
| Types + interfaces | ~2,000 LOC | 402 LOC | 80% |
| **Total** | **~53,000 LOC** | **~2,800 LOC** | **95%** |

What was cut: Neo4j/FalkorDB/Kuzu/Neptune drivers, cloud graph DB orchestration, Pydantic v2 model layer, Langchain integration, REST API surface, Docker config, multi-tenancy abstractions.

What remains: the temporal logic — entity resolution, relation invalidation, time-aware search.

---

## Install

```bash
pip install httpx  # only external dependency
```

Copy the `temporal/` folder into your project. For embeddings, bring any function that returns `list[float]`.

```
temporal/
├── types.py        # Data model (Episode, Entity, Relation, SearchResult)
├── interfaces.py   # Protocol definitions (LLMClient, Embedder, TemporalStore)
├── store.py        # SQLite implementation
├── llm_adapter.py  # OpenAI-compatible LLM client
├── prompts.py      # Extraction + resolution prompts
├── resolve.py      # Entity + relation resolution engine
├── retain.py       # Ingest pipeline
└── search.py       # Hybrid text + vector search
```

---

## Quick start

```python
import asyncio
from temporal import retain, search, SQLiteTemporalStore
from temporal.llm_adapter import OpenAICompatibleClient
from temporal.interfaces import Embedder

# Minimal embedder stub (replace with your embedding model)
class MyEmbedder:
    async def embed(self, text: str) -> list[float]:
        # Use OpenAI, Ollama nomic-embed-text, sentence-transformers, etc.
        ...

async def main():
    store = SQLiteTemporalStore("memory.db")
    llm = OpenAICompatibleClient(base_url="http://localhost:11434/v1", model="llama3.2")
    embedder = MyEmbedder()
    group_id = "user-123"  # partition per user/agent

    # Ingest facts
    result = await retain(
        content="Sarah is the CTO of Horizon Labs as of Q1 2025.",
        store=store,
        llm=llm,
        embedder=embedder,
        group_id=group_id,
    )
    print(f"Extracted {len(result.entities)} entities, {len(result.relations)} relations")

    # Later, a fact changes:
    await retain(
        content="Sarah left Horizon Labs in June 2025.",
        store=store,
        llm=llm,
        embedder=embedder,
        group_id=group_id,
    )

    # Search — old relation is invalidated, new one surfaces
    results = await search(
        query="Who leads Horizon Labs?",
        store=store,
        embedder=embedder,
        group_id=group_id,
    )
    for r in results.relations:
        print(f"{r.relation.source_entity_name} → {r.relation.name} → {r.relation.target_entity_name}")
        print(f"  fact: {r.relation.fact}")
        print(f"  valid_at: {r.relation.valid_at}  invalid_at: {r.relation.invalid_at}")

asyncio.run(main())
```

---

## How it works

### The temporal knowledge graph

Temporal stores three kinds of objects:

**Episodes** — the raw inputs (messages, documents, events):
```python
Episode(
    id="...",
    content="Sarah left Horizon Labs in June 2025.",
    episode_type=EpisodeType.message,
    reference_time="2025-06-15T00:00:00+00:00",
    group_id="user-123",
)
```

**Entities** — named things extracted from episodes:
```python
Entity(name="Sarah", entity_type=EntityType.person, summary="Executive, formerly CTO at Horizon Labs")
Entity(name="Horizon Labs", entity_type=EntityType.organization, summary="Tech company")
```

**Relations** — facts linking entities, with temporal validity:
```python
Relation(
    source_entity_name="Sarah",
    name="LEFT",
    target_entity_name="Horizon Labs",
    fact="Sarah left Horizon Labs.",
    valid_at="2025-06-15T00:00:00+00:00",
    invalid_at=None,  # still true
)
```

When a new fact contradicts an old one, the old relation gets `invalid_at` set and a new relation is created. The graph stays accurate — you can query what was true at any timestamp.

### Retain pipeline

```
input text
    ↓
Episode saved
    ↓
LLM extracts entities + relations from episode
    ↓
Resolve: match against existing entities (name + embedding similarity)
    ↓
Resolve: check if relation already exists (dedup or update)
    ↓
If contradicts existing fact → invalidate old relation, save new one
    ↓
Embed relation facts for vector search
    ↓
Save everything to SQLite
```

### Search

Hybrid retrieval — text match + embedding similarity, fused with RRF (Reciprocal Rank Fusion):

```python
results = await search(
    query="Sarah's role",
    store=store,
    embedder=embedder,
    group_id="user-123",
    filters=SearchFilters(
        valid_at_start="2024-01-01T00:00:00+00:00",
        valid_at_end="2025-01-01T00:00:00+00:00",
        include_invalidated=False,  # only facts still true in that window
    ),
    limit=10,
)
```

---

## Temporal filtering

Every query can be scoped to a point in time or a time window:

```python
from temporal import SearchFilters

# Facts that were true in 2024
filters = SearchFilters(
    valid_at_start="2024-01-01T00:00:00+00:00",
    valid_at_end="2024-12-31T00:00:00+00:00",
    include_invalidated=False,
    include_expired=False,
)

# Only look at specific relation types
filters = SearchFilters(relation_names=["WORKS_AT", "LEADS", "FOUNDED"])

# Filter by entity names
filters = SearchFilters(entity_names=["Sarah", "Alice"])
```

---

## SQLite schema

```sql
episodes       -- raw source content with reference timestamps
entities       -- named things with name embeddings
relations      -- facts between entities (valid_at, invalid_at, expired_at)
episodic_links -- provenance: which episode produced which entity
```

WAL mode enabled. Vector similarity computed in Python — handles up to ~10k relations without issue. For larger graphs, swap the `TemporalStore` protocol with a vector DB backend.

---

## Tests

```bash
# Unit tests (no LLM required)
python3 -m pytest tests/ -q --ignore=tests/test_e2e.py

# End-to-end test (requires an LLM endpoint)
OPENROUTER_API_KEY=sk-... \
  python3 -m pytest tests/test_e2e.py -v -s
```

155 tests covering types, entity resolution, relation invalidation, temporal filtering, hybrid search, retain pipeline.

---

## What was removed from Graphiti

Temporal is a targeted extraction of Graphiti's temporal logic, not a port of the full platform:

- **Graph database drivers** (Neo4j, FalkorDB, Kuzu, Neptune) — replaced with a single ~430 LOC SQLite store
- **Pydantic v2 model layer** — replaced with stdlib `dataclasses`
- **Langchain / LangGraph integration** — removed
- **REST API** — not included (this is a library, not a service)
- **Cloud infrastructure** (Zep Cloud, Docker compose) — removed
- **Multi-driver abstraction** — unnecessary when there's one store

The temporal validity model (valid_at, invalid_at, expired_at), the entity resolution algorithm, and the extraction prompts are preserved.

---

## Part of a suite

Temporal pairs naturally with:

- **[NanoLetta](https://github.com/DomLynch/NanoLetta)** — cognitive agent loop (Letta → 1.9k LOC). Wire Temporal in as a custom tool.
- **[Lucid](https://github.com/DomLynch/Lucid)** — semantic memory runtime (Hindsight → 2k LOC). `retain() / recall() / reflect()`.

---

## Requirements

- Python 3.11+
- `httpx` (for the built-in LLM adapter — omit if you bring your own)
- Any OpenAI-compatible LLM endpoint for extraction
- Any embedding function returning `list[float]`

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Acknowledgments

The temporal knowledge graph design, entity resolution logic, and valid_at/invalid_at model come from [Graphiti](https://github.com/getzep/graphiti) by Zep AI (Apache 2.0). Temporal is an independent extraction — not affiliated with Zep.
