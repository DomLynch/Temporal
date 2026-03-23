"""
temporal/retain.py — Episode ingest orchestrator.

This is the main entry point for adding knowledge to the temporal graph.
It orchestrates the full pipeline:

1. Create episode record
2. Retrieve prior episode context
3. Extract entities from episode content (via LLM)
4. Resolve entities against existing graph (dedup)
5. Extract relations between entities (via LLM)
6. Resolve relations (duplicate/contradiction detection)
7. Apply temporal invalidation for contradicted facts
8. Generate embeddings
9. Persist everything

Extracted from Graphiti's graphiti.py:add_episode() (250 LOC → ~350 LOC).
Stripped: communities, saga, typed edges, telemetry spans, driver routing.
"""

from __future__ import annotations

import logging

from temporal.interfaces import Embedder, LLMClient, TemporalStore
from temporal.llm_adapter import accumulate_usage, llm_extract
from temporal.prompts import (
    extract_entities_message,
    extract_entities_text,
    extract_entities_json,
    extract_relations,
)
from temporal.resolve import resolve_entities, resolve_relations
from temporal.types import (
    Entity,
    EntityType,
    Episode,
    EpisodeType,
    EpisodicLink,
    Relation,
    RetainResult,
    _now_iso,
)

_log = logging.getLogger("temporal.retain")


async def retain(
    content: str,
    group_id: str,
    name: str = "",
    source: str = "",
    episode_type: EpisodeType = EpisodeType.MESSAGE,
    reference_time: str | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    store: TemporalStore | None = None,
) -> RetainResult:
    """Ingest an episode into the temporal knowledge graph.

    This is the main entry point. Call this with any text — a conversation
    message, a document, structured data — and Temporal will extract
    entities, relations, and temporal validity, resolve against existing
    knowledge, and persist everything.

    Args:
        content: The text to process (conversation, document, etc.)
        group_id: Partition ID (e.g., "brain", "personal")
        name: Episode name/label
        source: Source identifier (e.g., "telegram", "whatsapp")
        episode_type: MESSAGE, TEXT, or JSON (affects extraction prompts)
        reference_time: When this happened (ISO datetime). Defaults to now.
        llm: LLM client for extraction and adjudication.
        embedder: For entity name and relation fact embeddings.
        store: Temporal store for persistence.

    Returns:
        RetainResult with counts and token usage.
    """
    if not content.strip():
        return RetainResult(success=False)

    result = RetainResult()
    total_usage = _zero_usage()

    # Step 1: Create episode
    episode = Episode(
        group_id=group_id,
        name=name or content[:80],
        content=content,
        source=source,
        episode_type=episode_type,
        reference_time=reference_time or _now_iso(),
    )
    result.episode_id = episode.id

    # Step 2: Retrieve prior episodes for context
    previous_episodes: list[Episode] = []
    if store:
        try:
            previous_episodes = await store.get_recent_episodes(
                group_id, limit=5, before=episode.reference_time
            )
        except Exception as exc:
            _log.debug("Failed to retrieve prior episodes: %s", exc)

    previous_context = _build_previous_context(previous_episodes)

    # Step 3: Extract entities via LLM
    entities, entity_usage = await _extract_entities(
        episode, previous_context, llm
    )
    total_usage = accumulate_usage(total_usage, entity_usage)
    result.entities_extracted = len(entities)

    if not entities:
        _log.debug("No entities extracted from episode %s", episode.id)
        # Still save the episode even if no entities found
        if store:
            await store.save_episode(episode)
        result.token_usage = total_usage
        return result

    # Step 4: Generate entity name embeddings
    if embedder and entities:
        try:
            name_texts = [e.name for e in entities]
            name_embeddings = await embedder.embed(name_texts)
            for entity, emb in zip(entities, name_embeddings):
                entity.name_embedding = emb
        except Exception as exc:
            _log.debug("Entity embedding failed: %s", exc)

    # Step 5: Resolve entities (dedup against existing)
    resolved_entities, _uuid_map, entity_resolve_usage = await resolve_entities(
        entities, episode, llm=llm, embedder=embedder, store=store
    )
    total_usage = accumulate_usage(total_usage, entity_resolve_usage)
    # Count entities that were merged into existing canonicals
    result.entities_resolved = sum(1 for old_id, new_id in _uuid_map.items() if old_id != new_id)

    # Step 6: Extract relations via LLM
    relations, relation_usage = await _extract_relations(
        episode, resolved_entities, previous_context, llm
    )
    total_usage = accumulate_usage(total_usage, relation_usage)
    result.relations_extracted = len(relations)

    # Step 7: Generate relation fact embeddings
    if embedder and relations:
        try:
            fact_texts = [r.fact for r in relations]
            fact_embeddings = await embedder.embed(fact_texts)
            for relation, emb in zip(relations, fact_embeddings):
                relation.fact_embedding = emb
        except Exception as exc:
            _log.debug("Relation embedding failed: %s", exc)

    # Step 8: Resolve relations (duplicate/contradiction/invalidation)
    resolved_relations, invalidated_relations, resolve_usage = await resolve_relations(
        relations, episode, llm=llm, embedder=embedder, store=store
    )
    total_usage = accumulate_usage(total_usage, resolve_usage)
    # Count relations that were resolved as duplicates (reused existing canonical)
    new_relation_ids = {r.id for r in relations}
    result.relations_resolved = sum(
        1 for r in resolved_relations if r.id not in new_relation_ids
    )
    result.relations_invalidated = len(invalidated_relations)

    # Step 9: Persist everything
    if store:
        await _persist(
            store, episode, resolved_entities, resolved_relations,
            invalidated_relations, group_id
        )

    result.token_usage = total_usage
    result.success = True

    _log.info(
        "Retained episode %s: %d entities (%d resolved), "
        "%d relations (%d invalidated), %d tokens",
        episode.id,
        result.entities_extracted,
        result.entities_resolved,
        result.relations_extracted,
        result.relations_invalidated,
        total_usage.get("total_tokens", 0),
    )

    return result


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

async def _extract_entities(
    episode: Episode,
    previous_context: str,
    llm: LLMClient | None,
) -> tuple[list[Entity], dict[str, int]]:
    """Extract entities from episode content via LLM."""
    if llm is None:
        return [], _zero_usage()

    context = {
        "episode_content": episode.content,
        "episode_type": episode.episode_type.value,
        "previous_context": previous_context,
        "source": episode.source,
    }

    # Select prompt based on episode type (per Graphiti kernel invariant)
    prompt_dispatch = {
        EpisodeType.MESSAGE: extract_entities_message,
        EpisodeType.TEXT: extract_entities_text,
        EpisodeType.JSON: extract_entities_json,
    }
    prompt_fn = prompt_dispatch.get(episode.episode_type, extract_entities_message)
    messages = prompt_fn(context)
    parsed, usage = await llm_extract(llm, messages, temperature=0.0)

    entities: list[Entity] = []
    raw_entities = parsed.get("entities", [])

    for raw in raw_entities:
        name = raw.get("name", "").strip()
        if not name:
            continue

        entity_type_str = raw.get("entity_type", "other").lower()
        try:
            entity_type = EntityType(entity_type_str)
        except ValueError:
            entity_type = EntityType.OTHER

        entity = Entity(
            group_id=episode.group_id,
            name=name,
            entity_type=entity_type,
            summary=raw.get("summary", ""),
            episode_ids=[episode.id],
        )
        entities.append(entity)

    _log.debug("Extracted %d entities from episode %s", len(entities), episode.id)
    return entities, usage


# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------

async def _extract_relations(
    episode: Episode,
    entities: list[Entity],
    previous_context: str,
    llm: LLMClient | None,
) -> tuple[list[Relation], dict[str, int]]:
    """Extract relations between entities via LLM."""
    if llm is None or not entities:
        return [], _zero_usage()

    # Build entity context for the prompt
    entity_list = [
        {"name": e.name, "type": e.entity_type.value}
        for e in entities
    ]

    # Build entity name→id lookup
    entity_map = {e.name.lower(): e for e in entities}

    context = {
        "episode_content": episode.content,
        "episode_type": episode.episode_type.value,
        "entities": entity_list,
        "previous_context": previous_context,
    }

    messages = extract_relations(context)
    parsed, usage = await llm_extract(llm, messages, temperature=0.0)

    relations: list[Relation] = []
    raw_relations = parsed.get("relations", [])

    for raw in raw_relations:
        source_name = raw.get("source", "").strip()
        target_name = raw.get("target", "").strip()
        fact = raw.get("fact", "").strip()

        if not source_name or not target_name or not fact:
            continue

        # Resolve entity names to IDs
        source_entity = entity_map.get(source_name.lower())
        target_entity = entity_map.get(target_name.lower())

        if not source_entity or not target_entity:
            _log.debug(
                "Skipping relation: entity not found (%s → %s)",
                source_name, target_name,
            )
            continue

        relation = Relation(
            group_id=episode.group_id,
            source_entity_id=source_entity.id,
            target_entity_id=target_entity.id,
            source_entity_name=source_entity.name,
            target_entity_name=target_entity.name,
            name=raw.get("relation_name", "related_to"),
            fact=fact,
            episode_ids=[episode.id],
            valid_at=raw.get("valid_at", episode.reference_time),
            invalid_at=raw.get("invalid_at"),
        )
        relations.append(relation)

    _log.debug("Extracted %d relations from episode %s", len(relations), episode.id)
    return relations, usage


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

async def _persist(
    store: TemporalStore,
    episode: Episode,
    entities: list[Entity],
    relations: list[Relation],
    invalidated: list[Relation],
    group_id: str,
) -> None:
    """Persist all extracted data to the store."""
    try:
        # Save episode
        await store.save_episode(episode)

        # Save entities
        if entities:
            await store.save_entities(entities)

        # Save episodic links (provenance)
        links = [
            EpisodicLink(
                episode_id=episode.id,
                entity_id=entity.id,
                group_id=group_id,
            )
            for entity in entities
        ]
        if links:
            await store.save_episodic_links(links)

        # Save resolved relations
        if relations:
            await store.save_relations(relations)

        # Save invalidated relations (updated with invalid_at/expired_at)
        # These are already persisted by _apply_temporal_invalidation in resolve.py,
        # but we save again to ensure consistency
        if invalidated:
            await store.save_relations(invalidated)

        _log.debug(
            "Persisted: 1 episode, %d entities, %d links, %d relations, %d invalidated",
            len(entities), len(links), len(relations), len(invalidated),
        )

    except Exception as exc:
        _log.error("Persistence failed for episode %s: %s", episode.id, exc)
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_previous_context(episodes: list[Episode]) -> str:
    """Build a summary of previous episodes for extraction context."""
    if not episodes:
        return ""

    lines = ["Previous context:"]
    for ep in episodes[-5:]:
        content_preview = ep.content[:200] if ep.content else ""
        lines.append(f"- [{ep.source or 'unknown'}] {content_preview}")

    return "\n".join(lines)


def _zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
