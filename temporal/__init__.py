"""
Temporal — Temporal knowledge graph for cognitive agents.

Extracted from Graphiti (53K LOC → ~2.7K LOC).
Knows when facts became true, when they changed, and why.

Core operations:
- retain(content) → extract entities + relations with temporal validity
- search(query) → retrieve with temporal filtering
"""

__version__ = "0.1.0"

from temporal.retain import retain
from temporal.search import search
from temporal.store import SQLiteTemporalStore
from temporal.types import (
    Entity,
    EntityType,
    Episode,
    EpisodeType,
    EpisodicLink,
    Relation,
    ResolutionVerdict,
    RetainResult,
    ResolveResult,
    SearchFilters,
    SearchResult,
    SearchResults,
)
from temporal.interfaces import (
    Embedder,
    LLMClient,
    Reranker,
    TemporalStore,
)

__all__ = [
    "__version__",
    # Public API
    "retain",
    "search",
    "SQLiteTemporalStore",
    # Types
    "Entity",
    "EntityType",
    "Episode",
    "EpisodeType",
    "EpisodicLink",
    "Relation",
    "ResolutionVerdict",
    "RetainResult",
    "ResolveResult",
    "SearchFilters",
    "SearchResult",
    "SearchResults",
    # Interfaces
    "Embedder",
    "LLMClient",
    "Reranker",
    "TemporalStore",
]
