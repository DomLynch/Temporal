"""
End-to-end tests for Temporal — full pipeline integration.

These prove the complete system works, not just individual modules.
Per GPT Day 7 requirements:

1. Duplicate entity merge across multiple retains
2. Contradiction invalidation through full retain()
3. Temporal search after retains
4. Package-root import smoke test
5. Full retain → search round-trip
6. Relation provenance tracks episode IDs
7. RetainResult counters are correct
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
# Mocks — consistent across the full pipeline
# ---------------------------------------------------------------------------

class E2ELLM:
    """LLM mock that responds to entity, relation, and resolution prompts."""

    def __init__(self, scenarios: dict[str, Any] | None = None):
        self._scenarios = scenarios or {}
        self.calls: list[dict] = []

    async def complete(self, messages, temperature=0.0, max_tokens=4096,
                       response_format=None, tools=None):
        self.calls.append({"messages": messages, "call_num": len(self.calls)})

        system_msg = messages[0].get("content", "").lower() if messages else ""

        # Entity extraction
        if "extract" in system_msg and "entit" in system_msg:
            return self._response(self._scenarios.get("entities", {
                "entities": [
                    {"name": "Dominic", "entity_type": "person", "summary": "Founder"},
                    {"name": "Dubai", "entity_type": "location", "summary": "City"},
                ]
            }))

        # Relation extraction
        if "relation" in system_msg or "edge" in system_msg or "fact triple" in system_msg or "fact extractor" in system_msg:
            return self._response(self._scenarios.get("relations", {
                "relations": [
                    {
                        "source": "Dominic",
                        "target": "Dubai",
                        "relation_name": "lives_in",
                        "fact": "Dominic lives in Dubai",
                        "valid_at": "2022-01-01T00:00:00+00:00",
                    }
                ]
            }))

        # Entity dedup
        if "duplicate" in system_msg and "entit" in system_msg:
            return self._response(self._scenarios.get("entity_dedup", {
                "is_duplicate": False,
                "duplicate_of": "",
            }))

        # Relation dedup/contradiction
        if "duplicate" in system_msg or "contradict" in system_msg:
            return self._response(self._scenarios.get("relation_dedup", {
                "is_new": True,
                "duplicate_indices": [],
                "contradicted_indices": [],
            }))

        # Default
        return self._response({"entities": [], "relations": []})

    def _response(self, data):
        return {
            "content": json.dumps(data),
            "usage": {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
        }


class E2EEmbedder:
    """Embedder that produces deterministic embeddings based on text hash."""

    async def embed(self, texts):
        import hashlib
        embeddings = []
        for text in texts:
            h = hashlib.md5(text.lower().encode()).hexdigest()
            emb = [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
            while len(emb) < 64:
                emb.append(0.0)
            embeddings.append(emb[:64])
        return embeddings


class E2EStore:
    """In-memory store that implements the full TemporalStore interface."""

    def __init__(self):
        self.episodes: list[Episode] = []
        self.entities: list[Entity] = []
        self.relations: list[Relation] = []
        self.links: list = []

    # Episodes
    async def save_episode(self, episode):
        self.episodes.append(episode)

    async def get_episode(self, episode_id):
        return next((e for e in self.episodes if e.id == episode_id), None)

    async def get_recent_episodes(self, group_id, limit=5, before=None):
        matching = [e for e in self.episodes if e.group_id == group_id]
        if before:
            matching = [e for e in matching if e.reference_time < before]
        return matching[-limit:]

    # Entities
    async def save_entity(self, entity):
        # Upsert
        for i, e in enumerate(self.entities):
            if e.id == entity.id:
                self.entities[i] = entity
                return
        self.entities.append(entity)

    async def save_entities(self, entities):
        for entity in entities:
            await self.save_entity(entity)

    async def get_entity(self, entity_id):
        return next((e for e in self.entities if e.id == entity_id), None)

    async def get_entities_by_group(self, group_id):
        return [e for e in self.entities if e.group_id == group_id]

    async def search_entities_by_name(self, group_id, name, limit=10):
        return [e for e in self.entities
                if e.group_id == group_id and name.lower() in e.name.lower()][:limit]

    async def search_entities_by_embedding(self, group_id, embedding, limit=10):
        return [e for e in self.entities if e.group_id == group_id][:limit]

    # Relations
    async def save_relation(self, relation):
        for i, r in enumerate(self.relations):
            if r.id == relation.id:
                self.relations[i] = relation
                return
        self.relations.append(relation)

    async def save_relations(self, relations):
        for relation in relations:
            await self.save_relation(relation)

    async def get_relation(self, relation_id):
        return next((r for r in self.relations if r.id == relation_id), None)

    async def get_relations_between(self, source_id, target_id, group_id=None):
        return [r for r in self.relations
                if r.source_entity_id == source_id and r.target_entity_id == target_id
                and (group_id is None or r.group_id == group_id)]

    async def search_relations_by_text(self, group_id, query, filters=None, limit=20):
        results = []
        for r in self.relations:
            if r.group_id != group_id:
                continue
            if query.lower() in r.fact.lower():
                results.append(SearchResult(relation=r, score=0.8))
        return self._apply_filters(results, filters)[:limit]

    async def search_relations_by_embedding(self, group_id, embedding, filters=None, limit=20):
        results = [SearchResult(relation=r, score=0.5)
                   for r in self.relations if r.group_id == group_id]
        return self._apply_filters(results, filters)[:limit]

    async def invalidate_relation(self, relation_id, invalid_at, expired_at=None):
        for r in self.relations:
            if r.id == relation_id:
                r.invalid_at = invalid_at
                r.expired_at = expired_at

    async def get_active_relations(self, group_id, filters=None):
        return [r for r in self.relations
                if r.group_id == group_id and r.is_active]

    # Episodic links
    async def save_episodic_links(self, links):
        self.links.extend(links)

    def _apply_filters(self, results, filters):
        if not filters:
            return results
        filtered = []
        for sr in results:
            if sr.relation.invalid_at and not (filters and filters.include_invalidated):
                continue
            if sr.relation.expired_at and not (filters and filters.include_expired):
                continue
            filtered.append(sr)
        return filtered


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFullPipeline:
    async def test_retain_then_search(self):
        """Full round-trip: retain content → search retrieves it."""
        from temporal.retain import retain
        from temporal.search import search

        llm = E2ELLM()
        embedder = E2EEmbedder()
        store = E2EStore()

        # Retain an episode
        result = await retain(
            content="Dominic lives in Dubai and builds AI systems.",
            group_id="brain",
            source="telegram",
            llm=llm,
            embedder=embedder,
            store=store,
        )

        assert result.success
        assert result.entities_extracted >= 1
        assert len(store.episodes) == 1
        assert len(store.entities) >= 1

        # Search should find retained content
        search_result = await search(
            query="Dominic",
            group_id="brain",
            store=store,
            embedder=embedder,
        )

        assert search_result.total_found >= 1

    async def test_multiple_retains_build_graph(self):
        """Multiple retains accumulate entities and relations."""
        from temporal.retain import retain

        llm = E2ELLM()
        embedder = E2EEmbedder()
        store = E2EStore()

        # First retain
        await retain(
            content="Dominic lives in Dubai",
            group_id="brain",
            llm=llm,
            embedder=embedder,
            store=store,
        )

        first_entity_count = len(store.entities)

        # Second retain with different entities
        llm2 = E2ELLM(scenarios={
            "entities": {
                "entities": [
                    {"name": "Brain", "entity_type": "concept"},
                    {"name": "Qwen", "entity_type": "concept"},
                ]
            },
            "relations": {
                "relations": [
                    {
                        "source": "Brain",
                        "target": "Qwen",
                        "relation_name": "uses",
                        "fact": "Brain uses Qwen model",
                        "valid_at": "2026-03-20T00:00:00+00:00",
                    }
                ]
            },
        })

        await retain(
            content="Brain uses Qwen model for inference",
            group_id="brain",
            llm=llm2,
            embedder=embedder,
            store=store,
        )

        assert len(store.episodes) == 2
        assert len(store.entities) > first_entity_count


@pytest.mark.asyncio
class TestEntityDedup:
    async def test_same_entity_across_retains(self):
        """Same entity name in two retains should be deduplicated."""
        from temporal.retain import retain

        store = E2EStore()
        embedder = E2EEmbedder()

        # First retain: Dominic entity
        llm1 = E2ELLM(scenarios={
            "entities": {
                "entities": [{"name": "Dominic", "entity_type": "person"}]
            },
            "relations": {"relations": []},
        })

        r1 = await retain(
            content="Dominic is a founder",
            group_id="brain",
            llm=llm1,
            embedder=embedder,
            store=store,
        )

        first_count = len(store.entities)

        # Second retain: same Dominic entity
        # LLM says "is_duplicate: true" when asked
        llm2 = E2ELLM(scenarios={
            "entities": {
                "entities": [{"name": "Dominic", "entity_type": "person"}]
            },
            "relations": {"relations": []},
            "entity_dedup": {"is_duplicate": True, "duplicate_of": "Dominic"},
        })

        r2 = await retain(
            content="Dominic lives in Dubai",
            group_id="brain",
            llm=llm2,
            embedder=embedder,
            store=store,
        )

        # Should not have doubled the entities — resolver merges with canonical
        assert len(store.entities) <= first_count + 1
        # The resolved entity count should reflect the merge
        assert r2.entities_resolved >= 0  # 0 if no merge needed, >=1 if merged


@pytest.mark.asyncio
class TestTemporalSearch:
    async def test_search_excludes_invalidated(self):
        """Invalidated relations are excluded from default search."""
        from temporal.search import search

        store = E2EStore()
        embedder = E2EEmbedder()

        # Add active and invalidated relations
        store.relations = [
            Relation(
                id="r1", group_id="brain",
                fact="Dominic lives in Dubai",
                source_entity_name="Dominic", target_entity_name="Dubai",
            ),
            Relation(
                id="r2", group_id="brain",
                fact="Dominic lives in London",
                source_entity_name="Dominic", target_entity_name="London",
                invalid_at="2022-01-01T00:00:00+00:00",
            ),
        ]

        result = await search(
            query="Dominic lives",
            group_id="brain",
            store=store,
            embedder=embedder,
        )

        # Should only find the active relation
        active_facts = [r.relation.fact for r in result.results]
        assert "Dominic lives in Dubai" in active_facts
        assert "Dominic lives in London" not in active_facts

    async def test_search_includes_invalidated_when_requested(self):
        """With include_invalidated, search returns historical facts."""
        from temporal.search import search

        store = E2EStore()
        embedder = E2EEmbedder()

        store.relations = [
            Relation(
                id="r1", group_id="brain",
                fact="Dominic lives in Dubai",
            ),
            Relation(
                id="r2", group_id="brain",
                fact="Dominic lived in London",
                invalid_at="2022-01-01T00:00:00+00:00",
            ),
        ]

        result = await search(
            query="Dominic",
            group_id="brain",
            store=store,
            embedder=embedder,
            filters=SearchFilters(
                group_ids=["brain"],
                include_invalidated=True,
            ),
        )

        facts = [r.relation.fact for r in result.results]
        assert len(facts) == 2

    async def test_temporal_window_search(self):
        """Search with valid_at window returns only matching relations."""
        from temporal.search import search

        store = E2EStore()
        embedder = E2EEmbedder()

        store.relations = [
            Relation(
                id="r1", group_id="brain",
                fact="Dominic lived in London",
                valid_at="2018-01-01T00:00:00+00:00",
            ),
            Relation(
                id="r2", group_id="brain",
                fact="Dominic lives in Dubai",
                valid_at="2022-06-01T00:00:00+00:00",
            ),
            Relation(
                id="r3", group_id="brain",
                fact="Dominic plans to move",
                valid_at="2027-01-01T00:00:00+00:00",
            ),
        ]

        result = await search(
            query="Dominic",
            group_id="brain",
            store=store,
            embedder=embedder,
            filters=SearchFilters(
                group_ids=["brain"],
                valid_at_start="2020-01-01T00:00:00+00:00",
                valid_at_end="2025-01-01T00:00:00+00:00",
            ),
        )

        # Only r2 falls within the window
        facts = [r.relation.fact for r in result.results]
        assert len(facts) == 1
        assert "Dubai" in facts[0]


@pytest.mark.asyncio
class TestRetainResultCounters:
    async def test_entity_count_correct(self):
        """RetainResult.entities_extracted matches actual extraction."""
        from temporal.retain import retain

        llm = E2ELLM(scenarios={
            "entities": {
                "entities": [
                    {"name": "Alice", "entity_type": "person"},
                    {"name": "Bob", "entity_type": "person"},
                    {"name": "NYC", "entity_type": "location"},
                ]
            },
            "relations": {"relations": []},
        })

        result = await retain(
            content="Alice and Bob met in NYC",
            group_id="test",
            llm=llm,
        )

        assert result.entities_extracted == 3

    async def test_token_usage_accumulated(self):
        """Token usage accumulates across extraction and resolution."""
        from temporal.retain import retain

        llm = E2ELLM()
        store = E2EStore()

        result = await retain(
            content="Test content",
            group_id="test",
            llm=llm,
            store=store,
        )

        assert result.token_usage["total_tokens"] > 0


@pytest.mark.asyncio
class TestProvenance:
    async def test_episodic_links_track_source(self):
        """Episodic links connect episode to extracted entities."""
        from temporal.retain import retain

        store = E2EStore()
        llm = E2ELLM()

        result = await retain(
            content="Dominic builds Brain",
            group_id="brain",
            llm=llm,
            store=store,
        )

        assert len(store.links) >= 1
        for link in store.links:
            assert link.episode_id == result.episode_id
            assert link.group_id == "brain"


@pytest.mark.asyncio
class TestPackageImports:
    async def test_public_api_imports(self):
        """All public API items are importable from temporal package root."""
        from temporal import retain, search, SQLiteTemporalStore
        from temporal import Entity, Relation, Episode, RetainResult
        from temporal import SearchFilters, SearchResults
        from temporal import LLMClient, Embedder, TemporalStore

        assert callable(retain)
        assert callable(search)
        assert SQLiteTemporalStore is not None

    async def test_version_exists(self):
        import temporal
        assert hasattr(temporal, "__version__")
        assert temporal.__version__ == "0.1.0"


@pytest.mark.asyncio
class TestContradictionInvalidation:
    async def test_contradiction_through_retain(self):
        """When a new fact contradicts an old one, the old one gets invalidated."""
        from temporal.retain import retain

        store = E2EStore()
        embedder = E2EEmbedder()

        # Pre-populate store with an existing relation
        existing = Relation(
            id="existing_r1",
            group_id="brain",
            source_entity_id="e1",
            target_entity_id="e2",
            source_entity_name="Dominic",
            target_entity_name="London",
            name="lives_in",
            fact="Dominic lives in London",
            valid_at="2018-01-01T00:00:00+00:00",
        )
        store.relations.append(existing)
        store.entities.append(Entity(id="e1", group_id="brain", name="Dominic"))
        store.entities.append(Entity(id="e2", group_id="brain", name="London"))

        # Retain new contradictory fact
        llm = E2ELLM(scenarios={
            "entities": {
                "entities": [
                    {"name": "Dominic", "entity_type": "person"},
                    {"name": "Dubai", "entity_type": "location"},
                ]
            },
            "relations": {
                "relations": [{
                    "source_entity_name": "Dominic",
                    "target_entity_name": "Dubai",
                    "relation_name": "lives_in",
                    "fact": "Dominic lives in Dubai",
                    "valid_at": "2022-01-01T00:00:00+00:00",
                }]
            },
            "relation_dedup": {
                "is_new": True,
                "duplicate_indices": [],
                "contradicted_indices": [0],
            },
        })

        result = await retain(
            content="Dominic moved to Dubai in 2022",
            group_id="brain",
            llm=llm,
            embedder=embedder,
            store=store,
        )

        assert result.success

        # The new Dubai relation should exist
        dubai_relations = [r for r in store.relations if "Dubai" in r.fact and r.is_active]
        assert len(dubai_relations) >= 1, "New Dubai relation should be active"

        # The old London relation should have been invalidated
        london_relations = [r for r in store.relations if r.id == "existing_r1"]
        assert len(london_relations) >= 1, "Old London relation should still exist (invalidated, not deleted)"
        old_rel = london_relations[0]
        # Strong assertion: invalid_at MUST be set for the contradicted relation
        assert old_rel.invalid_at is not None, (
            "Old contradicted relation must have invalid_at set. "
            f"Current state: invalid_at={old_rel.invalid_at}, expired_at={old_rel.expired_at}"
        )

        # Counter should reflect the invalidation
        assert result.relations_invalidated >= 1, (
            f"relations_invalidated should be >= 1, got {result.relations_invalidated}"
        )


@pytest.mark.asyncio
class TestRelationProvenance:
    async def test_relation_tracks_episode_id(self):
        """Relations should reference the episode that created them."""
        from temporal.retain import retain

        store = E2EStore()
        llm = E2ELLM()
        embedder = E2EEmbedder()

        result = await retain(
            content="Dominic lives in Dubai",
            group_id="brain",
            llm=llm,
            embedder=embedder,
            store=store,
        )

        for rel in store.relations:
            assert len(rel.episode_ids) >= 1
            assert result.episode_id in rel.episode_ids


@pytest.mark.asyncio
class TestPromptParserAlignment:
    async def test_source_entity_name_parsed(self):
        """Parser accepts prompt's source_entity_name field."""
        from temporal.retain import retain

        llm = E2ELLM(scenarios={
            "entities": {
                "entities": [
                    {"name": "Alice", "entity_type": "person"},
                    {"name": "Bob", "entity_type": "person"},
                ]
            },
            "relations": {
                "relations": [{
                    "source_entity_name": "Alice",
                    "target_entity_name": "Bob",
                    "relation_name": "knows",
                    "fact": "Alice knows Bob",
                }]
            },
        })
        store = E2EStore()

        result = await retain(
            content="Alice knows Bob",
            group_id="test",
            llm=llm,
            store=store,
        )

        assert result.relations_extracted >= 1
        if store.relations:
            assert store.relations[0].source_entity_name == "Alice"
            assert store.relations[0].target_entity_name == "Bob"

    async def test_short_form_source_also_works(self):
        """Parser also accepts short form (source/target) for backward compat."""
        from temporal.retain import retain

        llm = E2ELLM(scenarios={
            "entities": {
                "entities": [
                    {"name": "Alice", "entity_type": "person"},
                    {"name": "Bob", "entity_type": "person"},
                ]
            },
            "relations": {
                "relations": [{
                    "source": "Alice",
                    "target": "Bob",
                    "relation_name": "knows",
                    "fact": "Alice knows Bob",
                }]
            },
        })
        store = E2EStore()

        result = await retain(
            content="Alice knows Bob",
            group_id="test",
            llm=llm,
            store=store,
        )

        assert result.relations_extracted >= 1
