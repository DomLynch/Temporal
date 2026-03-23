"""
temporal/types.py — Core data types for temporal knowledge graph.

Extracted from Graphiti's nodes.py (1,081 LOC) + edges.py (1,036 LOC).
Stripped to essential types with temporal semantics.

Key types:
- Episode: a raw input event (conversation, document, etc.)
- Entity: a canonical thing (person, org, concept)
- Relation: a fact connecting two entities with temporal validity
- SearchResult: ranked retrieval output
- SearchFilters: temporal + partition filtering

Temporal semantics (the core value):
- valid_at:   when the fact became true
- invalid_at: when the fact stopped being true
- expired_at: when the system retired the relation from active truth
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_id_counter = 0

def _new_id() -> str:
    """Generate a short unique ID."""
    global _id_counter
    _id_counter += 1
    return hashlib.sha256(
        f"{datetime.now(timezone.utc).isoformat()}-{_id_counter}-{id(object())}".encode()
    ).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EpisodeType(Enum):
    """Source type of an episode — determines prompt selection."""
    MESSAGE = "message"       # Conversational message
    TEXT = "text"             # Document / article / note
    JSON = "json"            # Structured data


class EntityType(Enum):
    """Classification of an entity."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    DATE = "date"
    OTHER = "other"


class ResolutionVerdict(Enum):
    """LLM adjudication result for relation resolution."""
    NEW = "new"                 # Genuinely new fact
    DUPLICATE = "duplicate"     # Same fact already exists
    CONTRADICTS = "contradicts" # Contradicts an existing fact
    UPDATE = "update"           # Refines/updates an existing fact


# ---------------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A raw input event — the source material for extraction.

    Episodes are the atomic unit of ingestion. Each episode produces
    zero or more entities and relations.
    """
    id: str = field(default_factory=_new_id)
    group_id: str = ""
    name: str = ""
    content: str = ""
    source: str = ""  # e.g., "telegram", "whatsapp", "document"
    episode_type: EpisodeType = EpisodeType.MESSAGE
    reference_time: str = field(default_factory=_now_iso)
    created_at: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """A canonical entity in the knowledge graph.

    Entities are deduplicated — "Dominic", "Dom", "the founder" should
    resolve to one canonical Entity. The name_embedding enables
    semantic matching during dedup.
    """
    id: str = field(default_factory=_new_id)
    group_id: str = ""
    name: str = ""
    entity_type: EntityType = EntityType.OTHER
    summary: str = ""
    name_embedding: list[float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    episode_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)


@dataclass
class Relation:
    """A fact connecting two entities with temporal validity.

    This is the core temporal type — the reason Temporal exists.

    Temporal fields:
    - valid_at:   when the fact became true in the real world
    - invalid_at: when the fact stopped being true (set during contradiction resolution)
    - expired_at: when the system retired this relation (may differ from invalid_at)

    A relation with invalid_at=None is currently believed true.
    A relation with invalid_at set is a historical fact.
    """
    id: str = field(default_factory=_new_id)
    group_id: str = ""
    source_entity_id: str = ""
    target_entity_id: str = ""
    source_entity_name: str = ""
    target_entity_name: str = ""
    name: str = ""          # relation type as string, e.g., "lives_in", "founded"
    fact: str = ""          # human-readable fact text
    fact_embedding: list[float] | None = None
    episode_ids: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    # Temporal fields — the core value
    valid_at: str | None = None     # ISO datetime: when fact became true
    invalid_at: str | None = None   # ISO datetime: when fact stopped being true
    expired_at: str | None = None   # ISO datetime: when system retired this relation

    created_at: str = field(default_factory=_now_iso)

    @property
    def is_active(self) -> bool:
        """A relation is active if not invalidated and not expired."""
        return self.invalid_at is None and self.expired_at is None


@dataclass
class EpisodicLink:
    """Links an episode to the entities it mentions.

    Preserves provenance: which episode introduced which entity.
    """
    id: str = field(default_factory=_new_id)
    episode_id: str = ""
    entity_id: str = ""
    group_id: str = ""
    created_at: str = field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# Search Types
# ---------------------------------------------------------------------------

@dataclass
class SearchFilters:
    """Filters for temporal-aware retrieval.

    The key differentiator from Lucid: temporal filtering.
    """
    group_ids: list[str] | None = None
    entity_names: list[str] | None = None
    relation_names: list[str] | None = None

    # Temporal filters
    valid_at_start: str | None = None   # Only relations valid after this time
    valid_at_end: str | None = None     # Only relations valid before this time
    include_invalidated: bool = False   # Include relations that have been invalidated
    include_expired: bool = False       # Include relations that have been expired

    # Limits
    limit: int = 20


@dataclass
class SearchResult:
    """A single search result with score."""
    relation: Relation
    score: float = 0.0
    source: str = ""  # "text", "vector", "temporal", "entity_graph"


@dataclass
class SearchResults:
    """Aggregated search results."""
    results: list[SearchResult] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    total_found: int = 0

    @property
    def relations(self) -> list[Relation]:
        return [r.relation for r in self.results]


# ---------------------------------------------------------------------------
# Ingestion Results
# ---------------------------------------------------------------------------

@dataclass
class RetainResult:
    """Result of episode ingestion."""
    success: bool = True
    episode_id: str = ""
    entities_extracted: int = 0
    entities_resolved: int = 0      # Merged with existing
    relations_extracted: int = 0
    relations_resolved: int = 0     # Duplicates/contradictions handled
    relations_invalidated: int = 0  # Old facts marked invalid
    token_usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    })


@dataclass
class ResolveResult:
    """Result of entity or relation resolution."""
    verdict: ResolutionVerdict = ResolutionVerdict.NEW
    canonical_id: str = ""      # ID of the canonical entity/relation
    merged: bool = False        # Whether a merge occurred
    invalidated_ids: list[str] = field(default_factory=list)  # IDs of invalidated relations
