"""
temporal/search.py — Hybrid search with temporal filtering.

Combines multiple retrieval strategies and merges via Reciprocal Rank
Fusion (RRF). Search is part of the kernel — not a convenience layer —
because invalidation quality depends on search quality.

Retrieval channels:
1. Text search — substring matching on fact text
2. Vector search — cosine similarity on fact embeddings
3. Entity graph — relations connected to a specific entity
4. Temporal — prioritizes recently-valid facts

Fusion: RRF with k=60 (standard constant from the paper).

Extracted from Graphiti's search/ directory (3,312 LOC → ~380 LOC).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from temporal.interfaces import Embedder, Reranker, TemporalStore
from temporal.types import (
    Entity,
    Relation,
    SearchFilters,
    SearchResult,
    SearchResults,
)

_log = logging.getLogger("temporal.search")

# RRF constant (from the original RRF paper)
RRF_K = 60

# Default search limit
DEFAULT_LIMIT = 20


async def search(
    query: str,
    group_id: str,
    store: TemporalStore | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    filters: SearchFilters | None = None,
    limit: int = DEFAULT_LIMIT,
) -> SearchResults:
    """Hybrid search with temporal filtering.

    Runs multiple retrieval channels, merges via RRF, optionally
    reranks, and applies temporal filters.

    Args:
        query: Search query text.
        group_id: Partition ID.
        store: Temporal store for retrieval.
        embedder: For generating query embeddings (vector search).
        reranker: For cross-encoder reranking.
        filters: Temporal and partition filters.
        limit: Maximum results to return.

    Returns:
        SearchResults with ranked relations.
    """
    if not query.strip() or store is None:
        return SearchResults()

    filters = filters or SearchFilters(group_ids=[group_id])
    if filters.group_ids is None:
        filters.group_ids = [group_id]

    # Channel 1: Text search
    text_results = await _text_search(query, group_id, store, filters, limit * 2)

    # Channel 2: Vector search (if embedder available)
    vector_results: list[SearchResult] = []
    if embedder:
        vector_results = await _vector_search(
            query, group_id, store, embedder, filters, limit * 2
        )

    # Channel 3: Entity graph search
    entity_results = await _entity_graph_search(query, group_id, store, filters, limit)

    # Merge via RRF
    merged = _rrf_merge(
        [text_results, vector_results, entity_results],
        limit=limit * 2,
    )

    # Apply temporal filters
    merged = _apply_temporal_filters(merged, filters)

    # Rerank if available
    if reranker and merged:
        merged = await _rerank(query, merged, reranker, limit)

    # Final limit
    merged = merged[:limit]

    # Collect unique entities mentioned in results
    entities = await _collect_entities(merged, store, group_id)

    return SearchResults(
        results=merged,
        entities=entities,
        total_found=len(merged),
    )


async def search_for_resolution(
    query: str,
    group_id: str,
    store: TemporalStore,
    embedder: Embedder | None = None,
    exclude_ids: list[str] | None = None,
    limit: int = 10,
) -> list[SearchResult]:
    """Simplified search for resolution candidate finding.

    Used by resolve.py to find relations that might be duplicates
    or contradictions. Skips reranking for speed.

    Args:
        query: Fact text to search for.
        group_id: Partition ID.
        store: Temporal store.
        embedder: For vector search.
        exclude_ids: Relation IDs to exclude from results.
        limit: Maximum results.

    Returns:
        List of SearchResult, ranked by RRF.
    """
    exclude = set(exclude_ids or [])

    # Text search
    text_results = await _text_search(
        query, group_id, store, SearchFilters(group_ids=[group_id]), limit * 2
    )

    # Vector search
    vector_results: list[SearchResult] = []
    if embedder:
        vector_results = await _vector_search(
            query, group_id, store, embedder,
            SearchFilters(group_ids=[group_id]), limit * 2,
        )

    # Merge
    merged = _rrf_merge([text_results, vector_results], limit=limit * 2)

    # Filter out excluded IDs
    merged = [r for r in merged if r.relation.id not in exclude]

    return merged[:limit]


# ---------------------------------------------------------------------------
# Retrieval channels
# ---------------------------------------------------------------------------

async def _text_search(
    query: str,
    group_id: str,
    store: TemporalStore,
    filters: SearchFilters,
    limit: int,
) -> list[SearchResult]:
    """Channel 1: Text-based search."""
    try:
        results = await store.search_relations_by_text(
            group_id, query, filters=filters, limit=limit
        )
        for r in results:
            r.source = "text"
        return results
    except Exception as exc:
        _log.debug("Text search failed: %s", exc)
        return []


async def _vector_search(
    query: str,
    group_id: str,
    store: TemporalStore,
    embedder: Embedder,
    filters: SearchFilters,
    limit: int,
) -> list[SearchResult]:
    """Channel 2: Vector similarity search."""
    try:
        embeddings = await embedder.embed([query])
        if not embeddings or not embeddings[0]:
            return []

        results = await store.search_relations_by_embedding(
            group_id, embeddings[0], filters=filters, limit=limit
        )
        for r in results:
            r.source = "vector"
        return results
    except Exception as exc:
        _log.debug("Vector search failed: %s", exc)
        return []


async def _entity_graph_search(
    query: str,
    group_id: str,
    store: TemporalStore,
    filters: SearchFilters,
    limit: int,
) -> list[SearchResult]:
    """Channel 3: Entity-based graph traversal.

    Finds entities matching the query, then retrieves relations
    connected to those entities.
    """
    try:
        # Find entities matching the query
        entity_matches = await store.search_entities_by_name(
            group_id, query, limit=3
        )

        if not entity_matches:
            return []

        # Collect relations connected to matched entities
        results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for entity in entity_matches:
            # Get all active relations involving this entity
            active = await store.get_active_relations(group_id, filters)
            for rel in active:
                if rel.id in seen_ids:
                    continue
                if rel.source_entity_id == entity.id or rel.target_entity_id == entity.id:
                    results.append(SearchResult(
                        relation=rel,
                        score=0.5,  # Base score for graph match
                        source="entity_graph",
                    ))
                    seen_ids.add(rel.id)

                if len(results) >= limit:
                    break

        return results[:limit]
    except Exception as exc:
        _log.debug("Entity graph search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Fusion and filtering
# ---------------------------------------------------------------------------

def _rrf_merge(
    result_lists: list[list[SearchResult]],
    limit: int = DEFAULT_LIMIT,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion across multiple result lists.

    RRF score = sum(1 / (k + rank)) across all lists where the item appears.
    k = 60 (standard constant from the RRF paper).

    This is the same fusion formula used by Graphiti, Hindsight, and
    most modern hybrid search systems.
    """
    scores: dict[str, float] = defaultdict(float)
    relation_map: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, sr in enumerate(results):
            rid = sr.relation.id
            scores[rid] += 1.0 / (RRF_K + rank)
            # Keep the first (highest-quality) SearchResult for each relation
            if rid not in relation_map:
                relation_map[rid] = sr

    # Sort by RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda rid: scores[rid], reverse=True)

    merged: list[SearchResult] = []
    for rid in sorted_ids[:limit]:
        sr = relation_map[rid]
        sr.score = scores[rid]
        merged.append(sr)

    return merged


def _parse_dt(iso_str: str | None) -> datetime | None:
    """Parse ISO datetime string safely."""
    if not iso_str:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _apply_temporal_filters(
    results: list[SearchResult],
    filters: SearchFilters,
) -> list[SearchResult]:
    """Apply temporal filters to search results.

    Uses parsed datetimes for comparison, not raw strings.
    This is the core temporal differentiator — only Temporal does this.
    """
    filtered: list[SearchResult] = []

    # Pre-parse filter boundaries
    filter_start = _parse_dt(filters.valid_at_start)
    filter_end = _parse_dt(filters.valid_at_end)

    for sr in results:
        rel = sr.relation

        # Filter invalidated relations
        if rel.invalid_at and not filters.include_invalidated:
            continue

        # Filter expired relations
        if rel.expired_at and not filters.include_expired:
            continue

        # Filter by valid_at window (using parsed datetimes)
        rel_valid = _parse_dt(rel.valid_at)
        if filter_start and rel_valid:
            if rel_valid < filter_start:
                continue
        if filter_end and rel_valid:
            if rel_valid > filter_end:
                continue

        # Filter by relation names
        if filters.relation_names and rel.name:
            if rel.name not in filters.relation_names:
                continue

        # Filter by entity names
        if filters.entity_names:
            rel_entities = {rel.source_entity_name.lower(), rel.target_entity_name.lower()}
            filter_entities = {n.lower() for n in filters.entity_names}
            if not rel_entities & filter_entities:
                continue

        filtered.append(sr)

    return filtered


async def _rerank(
    query: str,
    results: list[SearchResult],
    reranker: Reranker,
    limit: int,
) -> list[SearchResult]:
    """Rerank results using a cross-encoder."""
    try:
        candidates = [sr.relation.fact for sr in results]
        ranked = await reranker.rerank(query, candidates, top_k=limit)

        reranked: list[SearchResult] = []
        for original_idx, score in ranked:
            if 0 <= original_idx < len(results):
                sr = results[original_idx]
                sr.score = score
                reranked.append(sr)

        return reranked
    except Exception as exc:
        _log.debug("Reranking failed: %s", exc)
        return results[:limit]


async def _collect_entities(
    results: list[SearchResult],
    store: TemporalStore,
    group_id: str,
) -> list[Entity]:
    """Collect unique entities mentioned in search results."""
    entity_ids: set[str] = set()
    for sr in results:
        if sr.relation.source_entity_id:
            entity_ids.add(sr.relation.source_entity_id)
        if sr.relation.target_entity_id:
            entity_ids.add(sr.relation.target_entity_id)

    if not entity_ids:
        return []

    entities: list[Entity] = []
    try:
        all_entities = await store.get_entities_by_group(group_id)
        entities = [e for e in all_entities if e.id in entity_ids]
    except Exception as exc:
        _log.debug("Entity collection failed: %s", exc)

    return entities
