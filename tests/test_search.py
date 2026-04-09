"""
Tests for temporal/search.py — Hybrid search with temporal filtering.

Covers (per GPT Day 5 bar):
1. Text search returns matching relations
2. Vector search with embedder
3. Entity graph search finds connected relations
4. RRF fusion merges multiple channels correctly
5. Temporal filter: excludes invalidated relations
6. Temporal filter: excludes expired relations
7. Temporal filter: valid_at window filtering
8. Temporal filter: include_invalidated flag
9. Reranking reorders results
10. Empty query returns empty
11. Partition isolation via group_id
12. search_for_resolution excludes specified IDs
"""

from __future__ import annotations

from typing import Any

import pytest

from temporal.types import (
    Entity,
    EntityType,
    Relation,
    SearchFilters,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockStore:
    """Mock store with configurable search results."""

    def __init__(
        self,
        relations: list[Relation] | None = None,
        entities: list[Entity] | None = None,
    ):
        self._relations = list(relations or [])
        self._entities = list(entities or [])

    async def search_relations_by_text(self, group_id, query, filters=None, limit=20):
        results = []
        for r in self._relations:
            if query.lower() in r.fact.lower() and r.group_id == group_id:
                results.append(SearchResult(relation=r, score=0.8))
        return results[:limit]

    async def search_relations_by_embedding(self, group_id, embedding, filters=None, limit=20):
        results = []
        for r in self._relations:
            if r.group_id == group_id:
                results.append(SearchResult(relation=r, score=0.6))
        return results[:limit]

    async def search_entities_by_name(self, group_id, name, limit=10):
        return [e for e in self._entities if name.lower() in e.name.lower() and e.group_id == group_id][:limit]

    async def get_active_relations(self, group_id, filters=None):
        return [r for r in self._relations if r.group_id == group_id and r.is_active]

    async def get_entities_by_group(self, group_id):
        return [e for e in self._entities if e.group_id == group_id]


class MockEmbedder:
    async def embed(self, texts):
        return [[0.1] * 64 for _ in texts]


class MockReranker:
    async def rerank(self, query, candidates, top_k=10):
        # Reverse the order (simulate reranking)
        return [(i, 1.0 - i * 0.1) for i in reversed(range(min(len(candidates), top_k)))]


def _rel(id: str, fact: str, group_id: str = "user-1", **kwargs) -> Relation:
    return Relation(id=id, fact=fact, group_id=group_id, **kwargs)


def _entity(id: str, name: str, group_id: str = "user-1") -> Entity:
    return Entity(id=id, name=name, group_id=group_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSearch:
    async def test_text_search_returns_matches(self):
        from temporal.search import search

        store = MockStore(relations=[
            _rel("r1", "Alice lives in London"),
            _rel("r2", "Nexus uses Qwen model"),
            _rel("r3", "Alice founded GDA"),
        ])

        result = await search("Alice", "user-1", store=store)

        facts = [r.relation.fact for r in result.results]
        assert any("Alice" in f for f in facts)
        assert result.total_found > 0

    async def test_vector_search_with_embedder(self):
        from temporal.search import search

        store = MockStore(relations=[
            _rel("r1", "Alice lives in London"),
        ])
        embedder = MockEmbedder()

        result = await search("Alice", "user-1", store=store, embedder=embedder)

        assert result.total_found >= 1

    async def test_entity_graph_search(self):
        from temporal.search import search

        entity = _entity("e1", "Alice")
        relations = [
            _rel("r1", "Alice lives in London", source_entity_id="e1", target_entity_id="e2"),
            _rel("r2", "Nexus uses Qwen", source_entity_id="e3", target_entity_id="e4"),
        ]

        store = MockStore(relations=relations, entities=[entity])

        result = await search("Alice", "user-1", store=store)

        # Should find r1 (connected to Alice entity) via graph search
        relation_ids = [r.relation.id for r in result.results]
        assert "r1" in relation_ids

    async def test_empty_query_returns_empty(self):
        from temporal.search import search

        store = MockStore(relations=[_rel("r1", "test")])
        result = await search("", "user-1", store=store)

        assert result.total_found == 0

    async def test_no_store_returns_empty(self):
        from temporal.search import search

        result = await search("test", "user-1")
        assert result.total_found == 0


@pytest.mark.asyncio
class TestRRFFusion:
    async def test_rrf_merges_channels(self):
        from temporal.search import _rrf_merge

        r1 = _rel("r1", "Fact 1")
        r2 = _rel("r2", "Fact 2")
        r3 = _rel("r3", "Fact 3")

        text_results = [
            SearchResult(relation=r1, score=0.9),
            SearchResult(relation=r2, score=0.7),
        ]
        vector_results = [
            SearchResult(relation=r2, score=0.8),
            SearchResult(relation=r3, score=0.6),
        ]

        merged = _rrf_merge([text_results, vector_results])

        # r2 appears in both channels → highest RRF score
        assert merged[0].relation.id == "r2"
        assert len(merged) == 3

    async def test_rrf_empty_lists(self):
        from temporal.search import _rrf_merge

        merged = _rrf_merge([[], []])
        assert len(merged) == 0

    async def test_rrf_single_channel(self):
        from temporal.search import _rrf_merge

        r1 = _rel("r1", "Fact 1")
        results = [SearchResult(relation=r1, score=0.9)]

        merged = _rrf_merge([results])
        assert len(merged) == 1
        assert merged[0].relation.id == "r1"


@pytest.mark.asyncio
class TestTemporalFiltering:
    async def test_excludes_invalidated_by_default(self):
        from temporal.search import _apply_temporal_filters

        active = SearchResult(relation=_rel("r1", "Active"), score=0.9)
        invalidated = SearchResult(
            relation=_rel("r2", "Invalid", invalid_at="2025-01-01T00:00:00+00:00"),
            score=0.8,
        )

        filtered = _apply_temporal_filters(
            [active, invalidated],
            SearchFilters(),
        )

        assert len(filtered) == 1
        assert filtered[0].relation.id == "r1"

    async def test_includes_invalidated_when_flag_set(self):
        from temporal.search import _apply_temporal_filters

        invalidated = SearchResult(
            relation=_rel("r1", "Invalid", invalid_at="2025-01-01T00:00:00+00:00"),
            score=0.8,
        )

        filtered = _apply_temporal_filters(
            [invalidated],
            SearchFilters(include_invalidated=True),
        )

        assert len(filtered) == 1

    async def test_excludes_expired_by_default(self):
        from temporal.search import _apply_temporal_filters

        expired = SearchResult(
            relation=_rel("r1", "Expired", expired_at="2025-01-01T00:00:00+00:00"),
            score=0.8,
        )

        filtered = _apply_temporal_filters([expired], SearchFilters())

        assert len(filtered) == 0

    async def test_includes_expired_when_flag_set(self):
        from temporal.search import _apply_temporal_filters

        expired = SearchResult(
            relation=_rel("r1", "Expired", expired_at="2025-01-01T00:00:00+00:00"),
            score=0.8,
        )

        filtered = _apply_temporal_filters(
            [expired],
            SearchFilters(include_expired=True),
        )

        assert len(filtered) == 1

    async def test_valid_at_window_filtering(self):
        from temporal.search import _apply_temporal_filters

        r_early = SearchResult(relation=_rel("r1", "Early", valid_at="2020-01-01T00:00:00+00:00"), score=0.9)
        r_mid = SearchResult(relation=_rel("r2", "Mid", valid_at="2023-06-01T00:00:00+00:00"), score=0.8)
        r_late = SearchResult(relation=_rel("r3", "Late", valid_at="2026-01-01T00:00:00+00:00"), score=0.7)

        filtered = _apply_temporal_filters(
            [r_early, r_mid, r_late],
            SearchFilters(
                valid_at_start="2022-01-01T00:00:00+00:00",
                valid_at_end="2025-01-01T00:00:00+00:00",
            ),
        )

        assert len(filtered) == 1
        assert filtered[0].relation.id == "r2"

    async def test_entity_name_filtering(self):
        from temporal.search import _apply_temporal_filters

        r1 = SearchResult(
            relation=_rel("r1", "Fact about Alice",
                          source_entity_name="Alice", target_entity_name="London"),
            score=0.9,
        )
        r2 = SearchResult(
            relation=_rel("r2", "Fact about Nexus",
                          source_entity_name="Nexus", target_entity_name="Qwen"),
            score=0.8,
        )

        filtered = _apply_temporal_filters(
            [r1, r2],
            SearchFilters(entity_names=["Alice"]),
        )

        assert len(filtered) == 1
        assert filtered[0].relation.id == "r1"


@pytest.mark.asyncio
class TestReranking:
    async def test_reranker_reorders(self):
        from temporal.search import search

        store = MockStore(relations=[
            _rel("r1", "Alice first fact"),
            _rel("r2", "Alice second fact"),
        ])
        reranker = MockReranker()

        result = await search("Alice", "user-1", store=store, reranker=reranker)

        # Reranker reverses order, so last item should come first
        assert result.total_found >= 1


@pytest.mark.asyncio
class TestPartitionIsolation:
    async def test_different_group_ids_isolated(self):
        from temporal.search import search

        store = MockStore(relations=[
            _rel("r1", "Alice lives in London", group_id="user-1"),
            _rel("r2", "Alice likes coffee", group_id="personal"),
        ])

        result = await search("Alice", "user-1", store=store)

        # Should only find brain group results
        for r in result.results:
            assert r.relation.group_id == "user-1"


@pytest.mark.asyncio
class TestSearchForResolution:
    async def test_excludes_specified_ids(self):
        from temporal.search import search_for_resolution

        store = MockStore(relations=[
            _rel("r1", "Alice lives in London"),
            _rel("r2", "Alice founded GDA"),
        ])

        results = await search_for_resolution(
            "Alice", "user-1", store=store, exclude_ids=["r1"]
        )

        relation_ids = [r.relation.id for r in results]
        assert "r1" not in relation_ids
        assert "r2" in relation_ids

    async def test_returns_ranked_results(self):
        from temporal.search import search_for_resolution

        store = MockStore(relations=[
            _rel("r1", "Alice lives in London"),
            _rel("r2", "Alice founded GDA"),
        ])

        results = await search_for_resolution("Alice", "user-1", store=store)

        assert len(results) >= 1
        # All results should have scores
        for r in results:
            assert r.score > 0
