"""
temporal/resolve.py — Entity and relation resolution with temporal invalidation.

This is the semantic heart of Temporal. It determines:
1. Whether a new entity is a duplicate of an existing one
2. Whether a new relation duplicates or contradicts existing ones
3. Which old facts should be invalidated when new facts arrive

Extracted from Graphiti's:
- utils/maintenance/node_operations.py (684 LOC)
- utils/maintenance/edge_operations.py (725 LOC)

Key invariants (from GPT kernel audit):
- Entity dedup uses search-assisted matching, not just string matching
- Relation resolution is two-stage: structural candidates → LLM adjudication
- Invalidation quality depends on search quality
- Duplicate reuse appends episode provenance, doesn't create new relations
- Temporal ordering: newer facts can invalidate older ones, AND older
  pre-existing facts can expire newly extracted ones
"""

from __future__ import annotations

import logging
from datetime import datetime

from temporal.interfaces import Embedder, LLMClient, TemporalStore
from temporal.llm_adapter import accumulate_usage, llm_extract
from temporal.prompts import resolve_entity_dedup, resolve_relation_dedup
from temporal.types import (
    Entity,
    Episode,
    Relation,
    ResolveResult,
    ResolutionVerdict,
    _now_iso,
)

_log = logging.getLogger("temporal.resolve")


# ---------------------------------------------------------------------------
# Entity Resolution
# ---------------------------------------------------------------------------

async def resolve_entities(
    new_entities: list[Entity],
    episode: Episode,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    store: TemporalStore | None = None,
) -> tuple[list[Entity], dict[str, str], dict[str, int]]:
    """Resolve new entities against existing ones in the graph.

    For each new entity:
    1. Search for existing entities by name (text + embedding)
    2. If candidates found, ask LLM if it's a duplicate
    3. If duplicate → reuse canonical entity, update provenance
    4. If new → keep as-is

    Args:
        new_entities: Entities extracted from the current episode.
        episode: The current episode (for context in LLM prompts).
        llm: LLM client for disambiguation.
        embedder: For generating name embeddings.
        store: For looking up existing entities.

    Returns:
        (resolved_entities, id_remap, token_usage)
        - resolved_entities: canonical entities (some may be existing)
        - id_remap: mapping from new entity IDs to canonical IDs
        - token_usage: accumulated LLM tokens
    """
    if not new_entities:
        return [], {}, _zero_usage()

    total_usage = _zero_usage()
    resolved: list[Entity] = []
    id_remap: dict[str, str] = {}  # new_id → canonical_id

    # Generate embeddings for new entities
    if embedder and new_entities:
        names = [e.name for e in new_entities]
        embeddings = await embedder.embed(names)
        for entity, emb in zip(new_entities, embeddings):
            entity.name_embedding = emb

    for new_entity in new_entities:
        if not new_entity.name.strip():
            continue

        # Search for existing candidates
        candidates = await _find_entity_candidates(
            new_entity, store
        )

        if not candidates or llm is None:
            # No candidates or no LLM — treat as new
            if episode.id not in new_entity.episode_ids:
                new_entity.episode_ids.append(episode.id)
            resolved.append(new_entity)
            id_remap[new_entity.id] = new_entity.id
            continue

        # Ask LLM to adjudicate
        result, usage = await _resolve_entity_with_llm(
            new_entity, candidates, episode, llm
        )
        total_usage = accumulate_usage(total_usage, usage)

        if result.verdict == ResolutionVerdict.DUPLICATE and result.canonical_id:
            # Reuse existing entity, update provenance
            canonical = next(
                (c for c in candidates if c.id == result.canonical_id),
                None,
            )
            if canonical:
                if episode.id not in canonical.episode_ids:
                    canonical.episode_ids.append(episode.id)
                resolved.append(canonical)
                id_remap[new_entity.id] = canonical.id
                _log.debug(
                    "Entity '%s' merged with existing '%s' (id=%s)",
                    new_entity.name, canonical.name, canonical.id,
                )
                continue

        # Not a duplicate — keep as new
        if episode.id not in new_entity.episode_ids:
            new_entity.episode_ids.append(episode.id)
        resolved.append(new_entity)
        id_remap[new_entity.id] = new_entity.id

    return resolved, id_remap, total_usage


async def _find_entity_candidates(
    entity: Entity,
    store: TemporalStore | None,
) -> list[Entity]:
    """Find existing entities that might be duplicates."""
    if store is None:
        return []

    candidates: list[Entity] = []
    seen_ids: set[str] = set()

    # Search by name (text match)
    try:
        name_matches = await store.search_entities_by_name(
            entity.group_id, entity.name, limit=5
        )
        for c in name_matches:
            if c.id not in seen_ids:
                candidates.append(c)
                seen_ids.add(c.id)
    except Exception as exc:
        _log.debug("Entity name search failed: %s", exc)

    # Search by embedding (semantic match)
    if entity.name_embedding:
        try:
            emb_matches = await store.search_entities_by_embedding(
                entity.group_id, entity.name_embedding, limit=5
            )
            for c in emb_matches:
                if c.id not in seen_ids:
                    candidates.append(c)
                    seen_ids.add(c.id)
        except Exception as exc:
            _log.debug("Entity embedding search failed: %s", exc)

    return candidates


async def _resolve_entity_with_llm(
    new_entity: Entity,
    candidates: list[Entity],
    episode: Episode,
    llm: LLMClient,
) -> tuple[ResolveResult, dict[str, int]]:
    """Use LLM to determine if new entity duplicates an existing one."""
    context = {
        "episode_content": episode.content,
        "new_entity": {
            "name": new_entity.name,
            "type": new_entity.entity_type.value,
        },
        "existing_entities": [
            {"name": c.name, "type": c.entity_type.value, "id": c.id}
            for c in candidates
        ],
    }

    messages = resolve_entity_dedup(context)
    parsed, usage = await llm_extract(llm, messages)

    is_duplicate = parsed.get("is_duplicate", False)
    duplicate_of_name = parsed.get("duplicate_of", "")

    if is_duplicate and duplicate_of_name:
        # Find the matching canonical entity
        canonical = next(
            (c for c in candidates if c.name.lower() == duplicate_of_name.lower()),
            None,
        )
        if canonical:
            return ResolveResult(
                verdict=ResolutionVerdict.DUPLICATE,
                canonical_id=canonical.id,
                merged=True,
            ), usage

    return ResolveResult(verdict=ResolutionVerdict.NEW), usage


# ---------------------------------------------------------------------------
# Relation Resolution
# ---------------------------------------------------------------------------

async def resolve_relations(
    new_relations: list[Relation],
    episode: Episode,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    store: TemporalStore | None = None,
) -> tuple[list[Relation], list[Relation], dict[str, int]]:
    """Resolve new relations against the existing graph.

    For each new relation:
    1. Check for exact duplicates (fast path — no LLM)
    2. Find related relations by endpoints + text search
    3. Ask LLM to classify: new / duplicate / contradicts
    4. Handle temporal invalidation for contradicted facts

    Args:
        new_relations: Relations extracted from the current episode.
        episode: The current episode.
        llm: LLM client for adjudication.
        embedder: For generating fact embeddings.
        store: For looking up existing relations.

    Returns:
        (resolved_relations, invalidated_relations, token_usage)
        - resolved_relations: canonical relations (some may be existing, reused)
        - invalidated_relations: old relations that were contradicted/expired
        - token_usage: accumulated LLM tokens
    """
    if not new_relations:
        return [], [], _zero_usage()

    total_usage = _zero_usage()

    # Step 1: Fast-path exact dedup within the batch
    new_relations = _dedup_within_batch(new_relations)

    # Step 2: Generate embeddings
    if embedder and new_relations:
        texts = [r.fact for r in new_relations]
        embeddings = await embedder.embed(texts)
        for relation, emb in zip(new_relations, embeddings):
            relation.fact_embedding = emb

    resolved: list[Relation] = []
    invalidated: list[Relation] = []

    for new_rel in new_relations:
        result, inv, usage = await _resolve_single_relation(
            new_rel, episode, llm, store
        )
        total_usage = accumulate_usage(total_usage, usage)
        resolved.append(result)
        invalidated.extend(inv)

    return resolved, invalidated, total_usage


async def _resolve_single_relation(
    new_rel: Relation,
    episode: Episode,
    llm: LLMClient | None,
    store: TemporalStore | None,
) -> tuple[Relation, list[Relation], dict[str, int]]:
    """Resolve a single relation against the graph.

    Returns:
        (resolved_relation, invalidated_relations, token_usage)
    """
    usage = _zero_usage()

    if store is None:
        # No store — just return the new relation
        if episode.id not in new_rel.episode_ids:
            new_rel.episode_ids.append(episode.id)
        return new_rel, [], usage

    # Step 1: Find existing relations between same endpoints
    endpoint_relations = await store.get_relations_between(
        new_rel.source_entity_id,
        new_rel.target_entity_id,
        group_id=new_rel.group_id,
    )

    # Step 2: Fast-path exact duplicate check
    normalized_fact = _normalize(new_rel.fact)
    for existing in endpoint_relations:
        if (
            existing.source_entity_id == new_rel.source_entity_id
            and existing.target_entity_id == new_rel.target_entity_id
            and _normalize(existing.fact) == normalized_fact
        ):
            # Exact duplicate — reuse, append provenance
            if episode.id not in existing.episode_ids:
                existing.episode_ids.append(episode.id)
            _log.debug("Exact duplicate: reusing relation %s", existing.id)
            return existing, [], usage

    # Step 3: Find broader invalidation candidates via text search
    invalidation_candidates: list[Relation] = []
    if store and new_rel.fact:
        try:
            search_results = await store.search_relations_by_text(
                new_rel.group_id, new_rel.fact, limit=10
            )
            # Exclude endpoint relations (already handled above)
            endpoint_ids = {r.id for r in endpoint_relations}
            invalidation_candidates = [
                sr.relation for sr in search_results
                if sr.relation.id not in endpoint_ids
            ]
        except Exception as exc:
            _log.debug("Invalidation search failed: %s", exc)

    # Step 4: No candidates at all — treat as new
    if not endpoint_relations and not invalidation_candidates:
        if episode.id not in new_rel.episode_ids:
            new_rel.episode_ids.append(episode.id)
        return new_rel, [], usage

    # Step 5: LLM adjudication
    if llm is None:
        # No LLM — treat as new (can't adjudicate)
        if episode.id not in new_rel.episode_ids:
            new_rel.episode_ids.append(episode.id)
        return new_rel, [], usage

    result, adj_usage = await _adjudicate_relation(
        new_rel, endpoint_relations, invalidation_candidates, llm
    )
    usage = accumulate_usage(usage, adj_usage)

    if result.verdict == ResolutionVerdict.DUPLICATE and result.canonical_id:
        # Reuse existing relation
        canonical = next(
            (r for r in endpoint_relations if r.id == result.canonical_id),
            None,
        )
        if canonical:
            if episode.id not in canonical.episode_ids:
                canonical.episode_ids.append(episode.id)
            _log.debug("Duplicate relation: reusing %s", canonical.id)
            return canonical, [], usage

    # Handle contradictions — invalidate old facts
    invalidated_rels: list[Relation] = []
    if result.invalidated_ids:
        all_candidates = endpoint_relations + invalidation_candidates
        invalidated_rels = _apply_temporal_invalidation(
            new_rel, result.invalidated_ids, all_candidates, store
        )

    # New or contradicting relation
    if episode.id not in new_rel.episode_ids:
        new_rel.episode_ids.append(episode.id)

    return new_rel, invalidated_rels, usage


async def _adjudicate_relation(
    new_rel: Relation,
    endpoint_relations: list[Relation],
    invalidation_candidates: list[Relation],
    llm: LLMClient,
) -> tuple[ResolveResult, dict[str, int]]:
    """Use LLM to classify a new relation against existing ones."""
    # Build context with continuous indexing (per Graphiti's pattern)
    existing_context = [
        {"idx": i, "fact": r.fact, "id": r.id}
        for i, r in enumerate(endpoint_relations)
    ]

    inv_offset = len(endpoint_relations)
    inv_context = [
        {"idx": inv_offset + i, "fact": r.fact, "id": r.id}
        for i, r in enumerate(invalidation_candidates)
    ]

    context = {
        "new_relation": {
            "fact": new_rel.fact,
            "source": new_rel.source_entity_name,
            "target": new_rel.target_entity_name,
        },
        "existing_relations": existing_context,
        "invalidation_candidates": inv_context,
    }

    messages = resolve_relation_dedup(context)
    parsed, usage = await llm_extract(llm, messages, temperature=0.0)

    duplicate_indices = parsed.get("duplicate_indices", [])
    contradicted_indices = parsed.get("contradicted_indices", [])
    is_new = parsed.get("is_new", True)

    # Map indices back to relation IDs
    all_relations = endpoint_relations + invalidation_candidates

    # Handle duplicates
    if duplicate_indices and not is_new:
        for idx in duplicate_indices:
            if 0 <= idx < len(all_relations):
                return ResolveResult(
                    verdict=ResolutionVerdict.DUPLICATE,
                    canonical_id=all_relations[idx].id,
                    merged=True,
                ), usage

    # Handle contradictions
    invalidated_ids: list[str] = []
    for idx in contradicted_indices:
        if 0 <= idx < len(all_relations):
            invalidated_ids.append(all_relations[idx].id)

    if invalidated_ids:
        return ResolveResult(
            verdict=ResolutionVerdict.CONTRADICTS,
            invalidated_ids=invalidated_ids,
        ), usage

    return ResolveResult(verdict=ResolutionVerdict.NEW), usage


def _apply_temporal_invalidation(
    new_rel: Relation,
    invalidated_ids: list[str],
    all_candidates: list[Relation],
    store: TemporalStore | None,
) -> list[Relation]:
    """Apply temporal invalidation to contradicted relations.

    Temporal ordering rules (from Graphiti kernel audit):
    1. Newer fact invalidates older: set invalid_at = new_fact.valid_at
    2. Older pre-existing fact can expire new: set new_fact.expired_at
    3. Already-invalidated relations are skipped

    Returns list of relations that were invalidated.
    """
    invalidated: list[Relation] = []
    now_str = _now_iso()

    new_valid_at = _parse_dt(new_rel.valid_at)
    new_invalid_at = _parse_dt(new_rel.invalid_at)

    for candidate in all_candidates:
        if candidate.id not in invalidated_ids:
            continue

        old_valid_at = _parse_dt(candidate.valid_at)
        old_invalid_at = _parse_dt(candidate.invalid_at)

        # Skip if already invalidated before new fact became valid
        if (
            old_invalid_at is not None
            and new_valid_at is not None
            and old_invalid_at <= new_valid_at
        ):
            continue

        # Skip if new fact was invalid before old fact became valid
        if (
            old_valid_at is not None
            and new_invalid_at is not None
            and new_invalid_at <= old_valid_at
        ):
            continue

        # New fact invalidates old fact
        if (
            old_valid_at is not None
            and new_valid_at is not None
            and old_valid_at < new_valid_at
        ):
            candidate.invalid_at = new_rel.valid_at
            if candidate.expired_at is None:
                candidate.expired_at = now_str
            invalidated.append(candidate)
            _log.info(
                "Invalidated relation %s: '%s' (superseded by '%s')",
                candidate.id, candidate.fact[:60], new_rel.fact[:60],
            )
        # Old fact already existed and is newer — expire the new fact instead
        elif (
            old_valid_at is not None
            and new_valid_at is not None
            and new_valid_at < old_valid_at
        ):
            new_rel.expired_at = now_str
            _log.info(
                "New relation expired: '%s' (pre-existing newer fact '%s')",
                new_rel.fact[:60], candidate.fact[:60],
            )

    return invalidated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup_within_batch(relations: list[Relation]) -> list[Relation]:
    """Fast-path dedup of exact duplicates within a single batch."""
    seen: dict[tuple[str, str, str], Relation] = {}
    result: list[Relation] = []

    for rel in relations:
        key = (rel.source_entity_id, rel.target_entity_id, _normalize(rel.fact))
        if key not in seen:
            seen[key] = rel
            result.append(rel)

    if len(result) < len(relations):
        _log.debug("Deduped %d → %d relations within batch", len(relations), len(result))

    return result


def _normalize(text: str) -> str:
    """Normalize text for exact comparison."""
    return " ".join(text.lower().split())


def _parse_dt(dt_str: str | None) -> datetime | None:
    """Parse an ISO datetime string, returning None on failure."""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        return None


def _zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
