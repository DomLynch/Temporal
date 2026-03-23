"""
temporal/interfaces.py — Protocol interfaces for Temporal.

Five pluggable boundaries:
1. LLMClient — for entity/relation extraction and resolution adjudication
2. Embedder — for entity name + relation fact embeddings
3. Reranker — for search result reranking
4. TemporalStore — for graph persistence with temporal fields
5. LLMAdapter — bridge from NanoLetta's complete() to Graphiti-style generate_response()

All use Protocol (structural subtyping) — implementations don't need
to inherit, just match the signature.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from temporal.types import (
    Entity,
    Episode,
    EpisodicLink,
    Relation,
    SearchFilters,
    SearchResult,
)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """LLM client for extraction, resolution, and adjudication.

    Supports both simple completion (text in → text out) and
    structured output (text in → parsed model out).

    Can reuse NanoLetta's OpenAICompatibleClient via the LLMAdapter.
    """

    async def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a completion request.

        Returns:
            {"content": str, "tool_calls": list | None,
             "usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int}}
        """
        ...


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    """Generate embeddings for text.

    Used for:
    - Entity name embeddings (semantic dedup)
    - Relation fact embeddings (vector search)
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns list of embedding vectors."""
        ...


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

@runtime_checkable
class Reranker(Protocol):
    """Rerank search candidates by relevance to a query."""

    async def rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Rerank candidates against a query.

        Returns:
            List of (original_index, relevance_score) sorted by score descending.
        """
        ...


# ---------------------------------------------------------------------------
# Temporal Store
# ---------------------------------------------------------------------------

@runtime_checkable
class TemporalStore(Protocol):
    """Persistence layer for the temporal knowledge graph.

    Must support:
    - Episode CRUD
    - Entity CRUD with dedup lookup
    - Relation CRUD with temporal fields
    - Episodic links (provenance)
    - Text + vector search with temporal filtering
    - Endpoint-based relation lookup (for resolution)
    """

    # Episodes
    async def save_episode(self, episode: Episode) -> None: ...
    async def get_episode(self, episode_id: str) -> Episode | None: ...
    async def get_recent_episodes(
        self, group_id: str, limit: int = 5, before: str | None = None
    ) -> list[Episode]: ...

    # Entities
    async def save_entity(self, entity: Entity) -> None: ...
    async def save_entities(self, entities: list[Entity]) -> None: ...
    async def get_entity(self, entity_id: str) -> Entity | None: ...
    async def get_entities_by_group(self, group_id: str) -> list[Entity]: ...
    async def search_entities_by_name(
        self, group_id: str, name: str, limit: int = 10
    ) -> list[Entity]: ...
    async def search_entities_by_embedding(
        self, group_id: str, embedding: list[float], limit: int = 10
    ) -> list[Entity]: ...

    # Relations
    async def save_relation(self, relation: Relation) -> None: ...
    async def save_relations(self, relations: list[Relation]) -> None: ...
    async def get_relation(self, relation_id: str) -> Relation | None: ...
    async def get_relations_between(
        self, source_id: str, target_id: str, group_id: str | None = None
    ) -> list[Relation]: ...
    async def search_relations_by_text(
        self, group_id: str, query: str, filters: SearchFilters | None = None, limit: int = 20
    ) -> list[SearchResult]: ...
    async def search_relations_by_embedding(
        self, group_id: str, embedding: list[float], filters: SearchFilters | None = None, limit: int = 20
    ) -> list[SearchResult]: ...
    async def invalidate_relation(
        self, relation_id: str, invalid_at: str, expired_at: str | None = None
    ) -> None: ...

    # Episodic links
    async def save_episodic_links(self, links: list[EpisodicLink]) -> None: ...

    # Maintenance
    async def get_active_relations(
        self, group_id: str, filters: SearchFilters | None = None
    ) -> list[Relation]: ...
