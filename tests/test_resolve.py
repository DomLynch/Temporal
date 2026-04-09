"""
Tests for temporal/resolve.py — Entity and relation resolution.

Covers (per GPT Day 4 bar):
1. Entity dedup: canonical reuse vs new entity creation
2. Entity dedup: LLM-driven disambiguation
3. Entity dedup: embedding-based candidate search
4. Relation duplicate detection: exact match fast path
5. Relation duplicate detection: LLM-driven semantic dedup
6. Contradiction adjudication: LLM-driven verdicts
7. Temporal ordering: newer fact invalidates older
8. Temporal ordering: pre-existing newer fact expires new one
9. Episode provenance: duplicate reuse appends episode
10. Fast paths: exact-match short-circuit before LLM
11. Batch dedup: identical relations within same batch
12. No store/no LLM graceful degradation
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
    ResolveResult,
    ResolutionVerdict,
    SearchResult,
    _now_iso,
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns pre-configured JSON responses."""

    def __init__(self, response: dict[str, Any] | None = None):
        self._response = response or {}
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages, temperature=0.0, max_tokens=4096,
                       response_format=None, tools=None):
        self.calls.append({"messages": messages})
        return {
            "content": json.dumps(self._response),
            "usage": {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
        }


class MockEmbedder:
    async def embed(self, texts):
        return [[float(i) / max(len(texts), 1)] * 64 for i in range(len(texts))]


class MockStore:
    """Mock store with configurable entity/relation lookup."""

    def __init__(
        self,
        entities: list[Entity] | None = None,
        relations: list[Relation] | None = None,
    ):
        self._entities = list(entities or [])
        self._relations = list(relations or [])
        self.saved_entities: list[Entity] = []
        self.saved_relations: list[Relation] = []
        self.invalidated: list[tuple[str, str, str | None]] = []

    async def search_entities_by_name(self, group_id, name, limit=10):
        return [e for e in self._entities if name.lower() in e.name.lower()][:limit]

    async def search_entities_by_embedding(self, group_id, embedding, limit=10):
        return self._entities[:limit]

    async def get_relations_between(self, source_id, target_id, group_id=None):
        return [
            r for r in self._relations
            if r.source_entity_id == source_id and r.target_entity_id == target_id
        ]

    async def search_relations_by_text(self, group_id, query, filters=None, limit=20):
        results = []
        for r in self._relations:
            if query.lower() in r.fact.lower():
                results.append(SearchResult(relation=r, score=0.8))
        return results[:limit]

    async def invalidate_relation(self, relation_id, invalid_at, expired_at=None):
        self.invalidated.append((relation_id, invalid_at, expired_at))


def _episode(content="Test episode") -> Episode:
    return Episode(
        id="ep_001", group_id="user-1", content=content,
        episode_type=EpisodeType.MESSAGE,
    )


# ---------------------------------------------------------------------------
# Entity Resolution Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEntityResolution:
    async def test_new_entity_no_candidates(self):
        """No existing entities → treat as new."""
        from temporal.resolve import resolve_entities

        entities = [Entity(id="e1", name="Alice", group_id="user-1")]
        store = MockStore(entities=[])

        resolved, remap, usage = await resolve_entities(
            entities, _episode(), store=store
        )

        assert len(resolved) == 1
        assert resolved[0].id == "e1"
        assert remap["e1"] == "e1"

    async def test_entity_dedup_via_llm(self):
        """LLM identifies entity as duplicate of existing one."""
        from temporal.resolve import resolve_entities

        existing = Entity(id="existing_1", name="Alexandra Chen", group_id="user-1")
        new = Entity(id="new_1", name="Alex", group_id="user-1")

        llm = MockLLM({"is_duplicate": True, "duplicate_of": "Alexandra Chen", "best_name": "Alexandra Chen"})
        store = MockStore(entities=[existing])

        resolved, remap, usage = await resolve_entities(
            [new], _episode(), llm=llm, store=store
        )

        assert len(resolved) == 1
        assert resolved[0].id == "existing_1"  # Reused canonical
        assert remap["new_1"] == "existing_1"

    async def test_entity_not_duplicate(self):
        """LLM says entities are different."""
        from temporal.resolve import resolve_entities

        existing = Entity(id="existing_1", name="London", group_id="user-1", entity_type=EntityType.LOCATION)
        new = Entity(id="new_1", name="Alice", group_id="user-1", entity_type=EntityType.PERSON)

        llm = MockLLM({"is_duplicate": False, "duplicate_of": "", "best_name": "Alice"})
        store = MockStore(entities=[existing])

        resolved, remap, usage = await resolve_entities(
            [new], _episode(), llm=llm, store=store
        )

        assert len(resolved) == 1
        assert resolved[0].id == "new_1"  # Kept as new
        assert remap["new_1"] == "new_1"

    async def test_entity_provenance_on_dedup(self):
        """When entity is deduplicated, episode ID is added to existing entity."""
        from temporal.resolve import resolve_entities

        existing = Entity(id="existing_1", name="Alexandra", group_id="user-1", episode_ids=["ep_old"])
        new = Entity(id="new_1", name="Alex", group_id="user-1")

        llm = MockLLM({"is_duplicate": True, "duplicate_of": "Alexandra", "best_name": "Alexandra"})
        store = MockStore(entities=[existing])

        resolved, _, _ = await resolve_entities(
            [new], _episode(), llm=llm, store=store
        )

        assert "ep_001" in resolved[0].episode_ids
        assert "ep_old" in resolved[0].episode_ids

    async def test_entity_with_embeddings(self):
        """Embeddings are generated for new entities."""
        from temporal.resolve import resolve_entities

        new = Entity(id="new_1", name="Test", group_id="user-1")
        embedder = MockEmbedder()
        store = MockStore(entities=[])

        resolved, _, _ = await resolve_entities(
            [new], _episode(), embedder=embedder, store=store
        )

        assert resolved[0].name_embedding is not None

    async def test_no_store_no_llm(self):
        """Without store or LLM, all entities are treated as new."""
        from temporal.resolve import resolve_entities

        entities = [
            Entity(id="e1", name="Alice", group_id="user-1"),
            Entity(id="e2", name="London", group_id="user-1"),
        ]

        resolved, remap, usage = await resolve_entities(entities, _episode())

        assert len(resolved) == 2
        assert usage["total_tokens"] == 0

    async def test_empty_name_skipped(self):
        """Entities with empty names are skipped."""
        from temporal.resolve import resolve_entities

        entities = [
            Entity(id="e1", name="", group_id="user-1"),
            Entity(id="e2", name="Alice", group_id="user-1"),
        ]

        resolved, _, _ = await resolve_entities(entities, _episode(), store=MockStore())

        assert len(resolved) == 1
        assert resolved[0].name == "Alice"


# ---------------------------------------------------------------------------
# Relation Resolution Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRelationResolution:
    async def test_exact_duplicate_fast_path(self):
        """Exact same fact + endpoints → reuse without LLM call."""
        from temporal.resolve import resolve_relations

        existing = Relation(
            id="r_existing", group_id="user-1",
            source_entity_id="e1", target_entity_id="e2",
            fact="Alice lives in London",
            episode_ids=["ep_old"],
        )
        new = Relation(
            id="r_new", group_id="user-1",
            source_entity_id="e1", target_entity_id="e2",
            fact="Alice lives in London",
        )

        store = MockStore(relations=[existing])

        resolved, invalidated, usage = await resolve_relations(
            [new], _episode(), store=store
        )

        assert len(resolved) == 1
        assert resolved[0].id == "r_existing"  # Reused
        assert "ep_001" in resolved[0].episode_ids  # Provenance added
        assert usage["total_tokens"] == 0  # No LLM call

    async def test_llm_detects_duplicate(self):
        """LLM identifies semantic duplicate (different wording)."""
        from temporal.resolve import resolve_relations

        existing = Relation(
            id="r_existing", group_id="user-1",
            source_entity_id="e1", target_entity_id="e2",
            fact="Alice resides in London",
        )
        new = Relation(
            id="r_new", group_id="user-1",
            source_entity_id="e1", target_entity_id="e2",
            fact="Alice lives in Paris",
        )

        llm = MockLLM({"duplicate_indices": [0], "contradicted_indices": [], "is_new": False})
        store = MockStore(relations=[existing])

        resolved, invalidated, usage = await resolve_relations(
            [new], _episode(), llm=llm, store=store
        )

        assert len(resolved) == 1
        assert resolved[0].id == "r_existing"  # Reused via LLM

    async def test_contradiction_invalidates_old(self):
        """New fact contradicts old → old gets invalidated."""
        from temporal.resolve import resolve_relations

        old = Relation(
            id="r_old", group_id="user-1",
            source_entity_id="e1", target_entity_id="e2",
            fact="Alice lives in Paris",
            valid_at="2018-01-01T00:00:00+00:00",
        )
        new = Relation(
            id="r_new", group_id="user-1",
            source_entity_id="e1", target_entity_id="e2",
            fact="Alice lives in London",
            valid_at="2022-01-01T00:00:00+00:00",
        )

        llm = MockLLM({"duplicate_indices": [], "contradicted_indices": [0], "is_new": True})
        store = MockStore(relations=[old])

        resolved, invalidated, usage = await resolve_relations(
            [new], _episode(), llm=llm, store=store
        )

        assert len(resolved) == 1
        assert resolved[0].id == "r_new"
        assert len(invalidated) == 1
        assert invalidated[0].id == "r_old"
        assert invalidated[0].invalid_at == "2022-01-01T00:00:00+00:00"

    async def test_newer_existing_expires_new(self):
        """Pre-existing newer fact causes new fact to be expired."""
        from temporal.resolve import resolve_relations

        existing_newer = Relation(
            id="r_newer", group_id="user-1",
            source_entity_id="e1", target_entity_id="e2",
            fact="Alice lives in London",
            valid_at="2022-01-01T00:00:00+00:00",
        )
        new_older = Relation(
            id="r_old_new", group_id="user-1",
            source_entity_id="e1", target_entity_id="e2",
            fact="Alice lives in Paris",
            valid_at="2018-01-01T00:00:00+00:00",
        )

        llm = MockLLM({"duplicate_indices": [], "contradicted_indices": [0], "is_new": True})
        store = MockStore(relations=[existing_newer])

        resolved, invalidated, usage = await resolve_relations(
            [new_older], _episode(), llm=llm, store=store
        )

        # The new (but older) fact should be expired, not the existing newer one
        assert resolved[0].expired_at is not None

    async def test_batch_dedup(self):
        """Identical relations in same batch are collapsed."""
        from temporal.resolve import resolve_relations

        r1 = Relation(id="r1", source_entity_id="e1", target_entity_id="e2", fact="Same fact", group_id="user-1")
        r2 = Relation(id="r2", source_entity_id="e1", target_entity_id="e2", fact="Same fact", group_id="user-1")

        resolved, _, _ = await resolve_relations([r1, r2], _episode())

        assert len(resolved) == 1

    async def test_no_store_treats_as_new(self):
        """Without store, all relations are new."""
        from temporal.resolve import resolve_relations

        rels = [
            Relation(id="r1", fact="Fact 1", group_id="user-1"),
            Relation(id="r2", fact="Fact 2", group_id="user-1"),
        ]

        resolved, invalidated, usage = await resolve_relations(rels, _episode())

        assert len(resolved) == 2
        assert len(invalidated) == 0

    async def test_provenance_on_new_relation(self):
        """New relations get episode ID added."""
        from temporal.resolve import resolve_relations

        rel = Relation(id="r1", fact="New fact", group_id="user-1")

        resolved, _, _ = await resolve_relations([rel], _episode())

        assert "ep_001" in resolved[0].episode_ids

    async def test_no_candidates_no_llm_call(self):
        """When no existing relations match, LLM is not called."""
        from temporal.resolve import resolve_relations

        rel = Relation(
            id="r1", group_id="user-1",
            source_entity_id="e1", target_entity_id="e2",
            fact="Brand new fact",
        )

        llm = MockLLM()
        store = MockStore(relations=[])  # No existing relations

        resolved, _, usage = await resolve_relations(
            [rel], _episode(), llm=llm, store=store
        )

        assert len(llm.calls) == 0  # No LLM call
        assert resolved[0].id == "r1"


# ---------------------------------------------------------------------------
# Temporal Invalidation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTemporalInvalidation:
    async def test_already_invalidated_skipped(self):
        """Relations already invalidated before new fact are skipped."""
        from temporal.resolve import _apply_temporal_invalidation

        new = Relation(
            fact="New", valid_at="2025-01-01T00:00:00+00:00",
        )
        old = Relation(
            id="old_1", fact="Old",
            valid_at="2020-01-01T00:00:00+00:00",
            invalid_at="2023-01-01T00:00:00+00:00",  # Already invalid before new
        )

        result = await _apply_temporal_invalidation(
            new, ["old_1"], [old], store=None,
        )

        assert len(result) == 0  # Skipped

    async def test_same_valid_at_no_invalidation(self):
        """Relations with same valid_at are not invalidated."""
        from temporal.resolve import _apply_temporal_invalidation

        new = Relation(fact="New", valid_at="2025-01-01T00:00:00+00:00")
        old = Relation(
            id="old_1", fact="Old",
            valid_at="2025-01-01T00:00:00+00:00",
        )

        result = await _apply_temporal_invalidation(
            new, ["old_1"], [old], store=None,
        )

        assert len(result) == 0  # Same time — no invalidation

    async def test_no_valid_at_no_crash(self):
        """Relations without valid_at don't cause errors."""
        from temporal.resolve import _apply_temporal_invalidation

        new = Relation(fact="New")
        old = Relation(id="old_1", fact="Old")

        result = await _apply_temporal_invalidation(
            new, ["old_1"], [old], store=None,
        )

        assert len(result) == 0  # Can't determine order
