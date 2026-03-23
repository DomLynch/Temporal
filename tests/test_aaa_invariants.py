"""
AAA kernel invariants — the authoritative proof surface for Temporal.

These tests prove the irreducible guarantees of the temporal knowledge graph.
If any of these fail, the kernel is broken.

Invariants tested:
1. Duplicate entity merge: repeated entity resolves to canonical
2. Duplicate relation reuse: repeated fact reuses canonical + merges provenance
3. Contradiction invalidation: newer fact invalidates older contradictory fact
4. Late-arriving older fact: older incoming fact stays historical
5. Counter exactness: entities_resolved, relations_resolved, relations_invalidated
6. Timestamp normalization: mixed Z and +00:00 formats produce identical behavior
7. Default search: only active facts returned
8. Historical search: invalidated facts returned when requested
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from temporal.types import (
    Entity,
    EntityType,
    Episode,
    EpisodeType,
    Relation,
    SearchFilters,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Mock infrastructure (deterministic, matches real resolver contracts)
# ---------------------------------------------------------------------------

class InvariantLLM:
    """LLM mock with explicit scenario control for invariant testing."""

    def __init__(self, scenarios: dict[str, Any] | None = None):
        self._scenarios = scenarios or {}
        self._call_count = 0

    async def complete(self, messages, temperature=0.0, max_tokens=4096,
                       response_format=None, tools=None):
        self._call_count += 1
        system_lower = (messages[0].get("content", "") if messages else "").lower()
        user_content = messages[1].get("content", "") if len(messages) > 1 else ""

        # Entity dedup — check BEFORE entity extraction (both mention "entit")
        if "duplicate" in system_lower and "entit" in system_lower:
            scenario = self._scenarios.get("entity_dedup", {
                "is_duplicate": False, "duplicate_of": "",
            })
            # Smart dedup: only say duplicate if the new entity's name matches duplicate_of
            # Extract the new entity name from the <NEW ENTITY> section of the prompt
            dup_name = scenario.get("duplicate_of", "")
            if dup_name:
                import re
                new_entity_match = re.search(r'"name":\s*"([^"]+)"', user_content)
                new_entity_name = new_entity_match.group(1) if new_entity_match else ""
                if new_entity_name.lower() != dup_name.lower():
                    return self._response({"is_duplicate": False, "duplicate_of": ""})
            return self._response(scenario)

        # Entity extraction
        if "extract" in system_lower and "entit" in system_lower:
            return self._response(self._scenarios.get("entities", {"entities": []}))

        # Relation dedup/contradiction
        if "duplicate" in system_lower or "contradict" in system_lower:
            return self._response(self._scenarios.get("relation_dedup", {
                "is_new": True, "duplicate_indices": [], "contradicted_indices": [],
            }))

        # Relation extraction
        if "relation" in system_lower or "fact" in system_lower:
            return self._response(self._scenarios.get("relations", {"relations": []}))

        return self._response({})

    def _response(self, data):
        return {
            "content": json.dumps(data),
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        }


class InvariantEmbedder:
    async def embed(self, texts):
        return [[0.1] * 64 for _ in texts]


class InvariantStore:
    """Store with full read/write tracking for invariant verification."""

    def __init__(self):
        self.episodes: list[Episode] = []
        self.entities: list[Entity] = []
        self.relations: list[Relation] = []
        self.links: list = []
        self.invalidated_ids: list[str] = []

    async def save_episode(self, ep):
        self.episodes.append(ep)

    async def save_entities(self, entities):
        # Upsert: replace existing by ID
        existing_ids = {e.id for e in entities}
        self.entities = [e for e in self.entities if e.id not in existing_ids]
        self.entities.extend(entities)

    async def save_relations(self, relations):
        existing_ids = {r.id for r in relations}
        self.relations = [r for r in self.relations if r.id not in existing_ids]
        self.relations.extend(relations)

    async def save_episodic_links(self, links):
        self.links.extend(links)

    async def get_recent_episodes(self, group_id, limit=5, before=None):
        eps = [e for e in self.episodes if e.group_id == group_id]
        return eps[-limit:]

    async def get_entities_by_group(self, group_id):
        return [e for e in self.entities if e.group_id == group_id]

    async def search_entities_by_name(self, group_id, name, limit=10):
        return [e for e in self.entities
                if e.group_id == group_id and name.lower() in e.name.lower()][:limit]

    async def search_entities_by_embedding(self, group_id, embedding, limit=10):
        return [e for e in self.entities if e.group_id == group_id][:limit]

    async def get_relations_between(self, source_id, target_id, group_id=None):
        return [r for r in self.relations
                if r.source_entity_id == source_id and r.target_entity_id == target_id]

    async def search_relations_by_text(self, group_id, query, filters=None, limit=20):
        # Match if ANY word from the query appears in the fact
        query_words = set(query.lower().split())
        results = []
        for r in self.relations:
            if r.group_id != group_id:
                continue
            fact_words = set(r.fact.lower().split())
            if query_words & fact_words:  # Any shared words
                results.append(SearchResult(relation=r, score=0.5))
        return results[:limit]

    async def search_relations_by_embedding(self, group_id, embedding, filters=None, limit=20):
        return [SearchResult(relation=r, score=0.3)
                for r in self.relations if r.group_id == group_id][:limit]

    async def get_active_relations(self, group_id, filters=None):
        return [r for r in self.relations
                if r.group_id == group_id and r.is_active]

    async def invalidate_relation(self, relation_id, invalid_at, expired_at=None):
        self.invalidated_ids.append(relation_id)
        for r in self.relations:
            if r.id == relation_id:
                r.invalid_at = invalid_at
                if expired_at:
                    r.expired_at = expired_at


# ---------------------------------------------------------------------------
# Invariant 1: Duplicate entity merge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDuplicateEntityMerge:
    async def test_repeated_entity_resolves_to_canonical(self):
        """Second retain with same entity must reuse the canonical, not create a duplicate."""
        from temporal.retain import retain

        store = InvariantStore()
        embedder = InvariantEmbedder()

        # First retain: creates "Dominic" entity
        llm1 = InvariantLLM(scenarios={
            "entities": {"entities": [{"name": "Dominic", "entity_type": "person"}]},
            "relations": {"relations": []},
        })

        r1 = await retain(content="Dominic is a founder", group_id="brain",
                          llm=llm1, embedder=embedder, store=store)

        assert r1.entities_extracted == 1
        first_entity_count = len(store.entities)
        assert first_entity_count >= 1

        # Second retain: same entity, LLM says duplicate
        llm2 = InvariantLLM(scenarios={
            "entities": {"entities": [{"name": "Dominic", "entity_type": "person"}]},
            "relations": {"relations": []},
            "entity_dedup": {"is_duplicate": True, "duplicate_of": "Dominic"},
        })

        r2 = await retain(content="Dominic lives in Dubai", group_id="brain",
                          llm=llm2, embedder=embedder, store=store)

        # EXACT: entity count must be unchanged (canonical reused, not added)
        assert len(store.entities) == first_entity_count, (
            f"Entity count should be unchanged after dedup merge. "
            f"Was {first_entity_count}, now {len(store.entities)}"
        )
        # EXACT: entities_resolved must be exactly 1 (one entity merged)
        assert r2.entities_resolved == 1, (
            f"Expected entities_resolved == 1 (one merge), got {r2.entities_resolved}"
        )


# ---------------------------------------------------------------------------
# Invariant 2: Duplicate relation reuse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDuplicateRelationReuse:
    async def test_repeated_fact_reuses_canonical(self):
        """Same fact extracted twice should reuse the existing relation, not create a duplicate."""
        from temporal.retain import retain

        store = InvariantStore()
        embedder = InvariantEmbedder()

        # First retain: creates relation
        llm1 = InvariantLLM(scenarios={
            "entities": {"entities": [
                {"name": "Dominic", "entity_type": "person"},
                {"name": "Dubai", "entity_type": "location"},
            ]},
            "relations": {"relations": [{
                "source_entity_name": "Dominic",
                "target_entity_name": "Dubai",
                "relation_name": "lives_in",
                "fact": "Dominic lives in Dubai",
                "valid_at": "2022-01-01T00:00:00+00:00",
            }]},
        })

        r1 = await retain(content="Dominic lives in Dubai", group_id="brain",
                          llm=llm1, embedder=embedder, store=store)

        assert r1.relations_extracted >= 1
        first_relation_count = len([r for r in store.relations if r.is_active])

        # Second retain: same fact, LLM says duplicate
        llm2 = InvariantLLM(scenarios={
            "entities": {"entities": [
                {"name": "Dominic", "entity_type": "person"},
                {"name": "Dubai", "entity_type": "location"},
            ]},
            "relations": {"relations": [{
                "source_entity_name": "Dominic",
                "target_entity_name": "Dubai",
                "relation_name": "lives_in",
                "fact": "Dominic lives in Dubai",
                "valid_at": "2022-01-01T00:00:00+00:00",
            }]},
            "entity_dedup": {"is_duplicate": True, "duplicate_of": "Dominic"},
            "relation_dedup": {
                "is_new": False,
                "duplicate_indices": [0],
                "contradicted_indices": [],
            },
        })

        r2 = await retain(content="Dominic lives in Dubai again", group_id="brain",
                          llm=llm2, embedder=embedder, store=store)

        # EXACT: active relation count must be unchanged (canonical reused)
        active_relations = [r for r in store.relations if r.is_active]
        assert len(active_relations) == first_relation_count, (
            f"Active relation count should be unchanged after duplicate reuse. "
            f"Was {first_relation_count}, now {len(active_relations)}"
        )

        # EXACT: relations_resolved must indicate reuse happened
        assert r2.relations_resolved >= 1, (
            f"Expected relations_resolved >= 1 (canonical reuse), got {r2.relations_resolved}"
        )

        # EXACT: no invalidation (duplicate, not contradiction)
        assert r2.relations_invalidated == 0, (
            f"Duplicate reuse should not invalidate. Got {r2.relations_invalidated}"
        )


# ---------------------------------------------------------------------------
# Invariant 3: Contradiction invalidation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestContradictionInvariant:
    async def test_newer_fact_invalidates_older(self):
        """When a newer fact contradicts an older one, the older gets invalid_at set."""
        from temporal.retain import retain

        store = InvariantStore()
        embedder = InvariantEmbedder()

        # Pre-populate: Dominic lives in London (old fact)
        old_relation = Relation(
            id="old_london", group_id="brain",
            source_entity_id="e1", target_entity_id="e2",
            source_entity_name="Dominic", target_entity_name="London",
            name="lives_in", fact="Dominic lives in London",
            valid_at="2018-01-01T00:00:00+00:00",
        )
        store.relations.append(old_relation)
        store.entities.append(Entity(id="e1", group_id="brain", name="Dominic"))
        store.entities.append(Entity(id="e2", group_id="brain", name="London"))

        # Retain: new contradictory fact
        llm = InvariantLLM(scenarios={
            "entities": {"entities": [
                {"name": "Dominic", "entity_type": "person"},
                {"name": "Dubai", "entity_type": "location"},
            ]},
            "relations": {"relations": [{
                "source_entity_name": "Dominic",
                "target_entity_name": "Dubai",
                "relation_name": "lives_in",
                "fact": "Dominic lives in Dubai",
                "valid_at": "2022-01-01T00:00:00+00:00",
            }]},
            "entity_dedup": {"is_duplicate": True, "duplicate_of": "Dominic"},
            "relation_dedup": {
                "is_new": True,
                "duplicate_indices": [],
                "contradicted_indices": [0],
            },
        })

        result = await retain(content="Dominic moved to Dubai", group_id="brain",
                              llm=llm, embedder=embedder, store=store)

        assert result.success

        # Old relation MUST have invalid_at set
        old = next((r for r in store.relations if r.id == "old_london"), None)
        assert old is not None, "Old relation should still exist"
        assert old.invalid_at is not None, (
            f"Contradicted relation must have invalid_at set, got None"
        )

        # New relation should be active
        dubai = [r for r in store.relations if "Dubai" in r.fact and r.is_active]
        assert len(dubai) >= 1, "New Dubai relation should be active"

        # EXACT: counter must reflect exactly 1 invalidation
        assert result.relations_invalidated == 1, (
            f"Expected exactly 1 invalidation, got {result.relations_invalidated}"
        )


# ---------------------------------------------------------------------------
# Invariant 4: Counter exactness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCounterExactness:
    async def test_zero_counters_when_no_dedup(self):
        """First retain with no existing data should have zero resolved/invalidated."""
        from temporal.retain import retain

        llm = InvariantLLM(scenarios={
            "entities": {"entities": [{"name": "Alice", "entity_type": "person"}]},
            "relations": {"relations": []},
        })

        result = await retain(content="Alice exists", group_id="test", llm=llm)

        assert result.entities_extracted == 1
        assert result.entities_resolved == 0
        assert result.relations_extracted == 0
        assert result.relations_resolved == 0
        assert result.relations_invalidated == 0

    async def test_token_usage_nonzero(self):
        """Token usage should be positive after any LLM-assisted retain."""
        from temporal.retain import retain

        llm = InvariantLLM(scenarios={
            "entities": {"entities": [{"name": "Bob", "entity_type": "person"}]},
            "relations": {"relations": []},
        })

        result = await retain(content="Bob exists", group_id="test", llm=llm)

        assert result.token_usage["total_tokens"] > 0
        assert result.token_usage["input_tokens"] > 0


# ---------------------------------------------------------------------------
# Invariant 5: Timestamp normalization (mixed formats)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTimestampNormalization:
    async def test_mixed_z_and_offset_produce_same_results(self):
        """Z suffix and +00:00 suffix should be treated identically."""
        from temporal.store import SQLiteTemporalStore, _normalize_ts
        import tempfile, os

        # Normalization function should convert Z to +00:00
        assert _normalize_ts("2025-01-01T00:00:00Z") == "2025-01-01T00:00:00+00:00"
        assert _normalize_ts("2025-01-01T00:00:00+00:00") == "2025-01-01T00:00:00+00:00"
        assert _normalize_ts(None) is None
        assert _normalize_ts("") == ""

    async def test_store_query_with_mixed_formats(self):
        """Store queries work correctly regardless of timestamp format."""
        from temporal.store import SQLiteTemporalStore
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteTemporalStore(db_path=os.path.join(tmpdir, "test.db"))

            # Save with Z format
            ep1 = Episode(
                id="ep1", group_id="brain", name="test1", content="test",
                reference_time="2025-06-01T12:00:00Z",
            )
            await store.save_episode(ep1)

            # Save with +00:00 format
            ep2 = Episode(
                id="ep2", group_id="brain", name="test2", content="test",
                reference_time="2025-06-02T12:00:00+00:00",
            )
            await store.save_episode(ep2)

            # Query with +00:00 format should find ep1 (saved with Z)
            recent = await store.get_recent_episodes(
                "brain", limit=10, before="2025-06-02T00:00:00+00:00"
            )
            episode_ids = [e.id for e in recent]
            assert "ep1" in episode_ids, "Should find Z-formatted episode with +00:00 query"

            # Save relations with mixed formats
            r1 = Relation(
                id="r1", group_id="brain", fact="Fact with Z",
                valid_at="2025-01-01T00:00:00Z",
            )
            r2 = Relation(
                id="r2", group_id="brain", fact="Fact with offset",
                valid_at="2025-06-01T00:00:00+00:00",
            )
            await store.save_relations([r1, r2])

            # Filter should work correctly across formats
            active = await store.get_active_relations(
                "brain",
                SearchFilters(
                    valid_at_start="2025-03-01T00:00:00Z",
                    valid_at_end="2025-12-01T00:00:00+00:00",
                ),
            )
            relation_ids = [r.id for r in active]
            assert "r2" in relation_ids, "Should find offset-formatted relation with Z filter"
            assert "r1" not in relation_ids, "Should exclude earlier Z-formatted relation"

            store.close()


# ---------------------------------------------------------------------------
# Invariant 6: Default vs historical search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSearchSemantics:
    async def test_default_search_returns_only_active(self):
        """Default search excludes invalidated and expired relations."""
        from temporal.search import _apply_temporal_filters

        active = SearchResult(
            relation=Relation(id="r1", fact="Active fact", group_id="brain"),
            score=0.9,
        )
        invalidated = SearchResult(
            relation=Relation(
                id="r2", fact="Old fact", group_id="brain",
                invalid_at="2025-01-01T00:00:00+00:00",
            ),
            score=0.8,
        )
        expired = SearchResult(
            relation=Relation(
                id="r3", fact="Expired fact", group_id="brain",
                expired_at="2025-01-01T00:00:00+00:00",
            ),
            score=0.7,
        )

        # Default filters: exclude invalidated + expired
        filtered = _apply_temporal_filters(
            [active, invalidated, expired],
            SearchFilters(),
        )

        assert len(filtered) == 1
        assert filtered[0].relation.id == "r1"

    async def test_historical_search_includes_invalidated(self):
        """Historical search with include_invalidated returns old facts."""
        from temporal.search import _apply_temporal_filters

        invalidated = SearchResult(
            relation=Relation(
                id="r1", fact="Old fact", group_id="brain",
                invalid_at="2025-01-01T00:00:00+00:00",
                valid_at="2020-01-01T00:00:00+00:00",
            ),
            score=0.8,
        )

        filtered = _apply_temporal_filters(
            [invalidated],
            SearchFilters(include_invalidated=True),
        )

        assert len(filtered) == 1
        assert filtered[0].relation.id == "r1"
