"""
Temporal — Temporal knowledge graph for cognitive agents.

Extracted from Graphiti (53K LOC → ~3.5K LOC).
Knows when facts became true, when they changed, and why.

Core operations:
- retain(episode) → extract entities + relations with temporal validity
- search(query) → retrieve with temporal filtering
- invalidate() → mark old facts as superseded
"""

__version__ = "0.1.0"

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
