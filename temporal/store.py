"""
temporal/store.py — SQLite persistence for temporal knowledge graph.

Implements the TemporalStore protocol with:
- Episode persistence
- Entity persistence with name search + embedding similarity
- Relation persistence with temporal fields (valid_at, invalid_at, expired_at)
- Endpoint-based relation lookup (for resolution)
- Text search + vector search with temporal filtering
- Episodic links (provenance)
- Partition isolation via group_id

Replaces Graphiti's:
- driver/neo4j/ (2,785 LOC)
- driver/falkordb/ (2,883 LOC)
- driver/kuzu/ (2,889 LOC)
- driver/neptune/ (2,816 LOC)
- driver/ top-level (1,736 LOC)
Total: ~13,109 LOC → ~430 LOC
"""

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

from temporal.types import (
    Entity,
    EntityType,
    Episode,
    EpisodeType,
    EpisodicLink,
    Relation,
    SearchFilters,
    SearchResult,
)

_log = logging.getLogger("temporal.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    episode_type TEXT NOT NULL DEFAULT 'message',
    reference_time TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    entity_type TEXT NOT NULL DEFAULT 'other',
    summary TEXT NOT NULL DEFAULT '',
    name_embedding TEXT NOT NULL DEFAULT '[]',
    attributes TEXT NOT NULL DEFAULT '{}',
    episode_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL DEFAULT '',
    source_entity_id TEXT NOT NULL DEFAULT '',
    target_entity_id TEXT NOT NULL DEFAULT '',
    source_entity_name TEXT NOT NULL DEFAULT '',
    target_entity_name TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    fact TEXT NOT NULL DEFAULT '',
    fact_embedding TEXT NOT NULL DEFAULT '[]',
    episode_ids TEXT NOT NULL DEFAULT '[]',
    attributes TEXT NOT NULL DEFAULT '{}',
    valid_at TEXT,
    invalid_at TEXT,
    expired_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS episodic_links (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    group_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_episodes_group ON episodes(group_id, reference_time);
CREATE INDEX IF NOT EXISTS idx_entities_group ON entities(group_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(group_id, name);
CREATE INDEX IF NOT EXISTS idx_relations_group ON relations(group_id);
CREATE INDEX IF NOT EXISTS idx_relations_endpoints ON relations(source_entity_id, target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_temporal ON relations(group_id, valid_at, invalid_at, expired_at);
CREATE INDEX IF NOT EXISTS idx_episodic_links ON episodic_links(episode_id, entity_id);
"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalize_ts(ts: str | None) -> str | None:
    """Normalize ISO timestamp to consistent format for lexicographic sorting.

    Ensures all timestamps use '+00:00' suffix (not 'Z') so SQL string
    comparisons produce chronologically correct results.
    """
    if not ts:
        return ts
    # Replace Z suffix with +00:00
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return ts


class SQLiteTemporalStore:
    """SQLite implementation of the TemporalStore protocol.

    Single-file, zero-dependency persistence for the temporal knowledge graph.
    Uses WAL mode for concurrent reads. Vector similarity computed in Python
    (acceptable for <10K relations).
    """

    def __init__(self, db_path: str | Path = "temporal.db") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    async def save_episode(self, episode: Episode) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO episodes
               (id, group_id, name, content, source, episode_type, reference_time, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (episode.id, episode.group_id, episode.name, episode.content,
             episode.source, episode.episode_type.value,
             _normalize_ts(episode.reference_time),
             _normalize_ts(episode.created_at), json.dumps(episode.metadata)),
        )
        conn.commit()

    async def get_episode(self, episode_id: str) -> Episode | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_episode(row)

    async def get_recent_episodes(
        self, group_id: str, limit: int = 5, before: str | None = None
    ) -> list[Episode]:
        conn = self._get_conn()
        if before:
            rows = conn.execute(
                """SELECT * FROM episodes WHERE group_id = ? AND reference_time < ?
                   ORDER BY reference_time DESC LIMIT ?""",
                (group_id, _normalize_ts(before), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM episodes WHERE group_id = ?
                   ORDER BY reference_time DESC LIMIT ?""",
                (group_id, limit),
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    async def save_entity(self, entity: Entity) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO entities
               (id, group_id, name, entity_type, summary, name_embedding, attributes, episode_ids, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entity.id, entity.group_id, entity.name, entity.entity_type.value,
             entity.summary, json.dumps(entity.name_embedding or []),
             json.dumps(entity.attributes), json.dumps(entity.episode_ids),
             entity.created_at),
        )
        conn.commit()

    async def save_entities(self, entities: list[Entity]) -> None:
        conn = self._get_conn()
        for entity in entities:
            conn.execute(
                """INSERT OR REPLACE INTO entities
                   (id, group_id, name, entity_type, summary, name_embedding, attributes, episode_ids, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entity.id, entity.group_id, entity.name, entity.entity_type.value,
                 entity.summary, json.dumps(entity.name_embedding or []),
                 json.dumps(entity.attributes), json.dumps(entity.episode_ids),
                 entity.created_at),
            )
        conn.commit()

    async def get_entity(self, entity_id: str) -> Entity | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_entity(row)

    async def get_entities_by_group(self, group_id: str) -> list[Entity]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM entities WHERE group_id = ?", (group_id,)
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    async def search_entities_by_name(
        self, group_id: str, name: str, limit: int = 10
    ) -> list[Entity]:
        conn = self._get_conn()
        name_lower = name.lower()
        rows = conn.execute(
            """SELECT * FROM entities WHERE group_id = ? AND LOWER(name) LIKE ?
               LIMIT ?""",
            (group_id, f"%{name_lower}%", limit),
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    async def search_entities_by_embedding(
        self, group_id: str, embedding: list[float], limit: int = 10
    ) -> list[Entity]:
        """Find entities by name embedding similarity."""
        all_entities = await self.get_entities_by_group(group_id)
        scored = []
        for ent in all_entities:
            if ent.name_embedding:
                sim = _cosine_similarity(embedding, ent.name_embedding)
                scored.append((ent, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ent for ent, _ in scored[:limit]]

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    async def save_relation(self, relation: Relation) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO relations
               (id, group_id, source_entity_id, target_entity_id, source_entity_name,
                target_entity_name, name, fact, fact_embedding, episode_ids, attributes,
                valid_at, invalid_at, expired_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (relation.id, relation.group_id, relation.source_entity_id,
             relation.target_entity_id, relation.source_entity_name,
             relation.target_entity_name, relation.name, relation.fact,
             json.dumps(relation.fact_embedding or []),
             json.dumps(relation.episode_ids), json.dumps(relation.attributes),
             _normalize_ts(relation.valid_at), _normalize_ts(relation.invalid_at),
             _normalize_ts(relation.expired_at), _normalize_ts(relation.created_at)),
        )
        conn.commit()

    async def save_relations(self, relations: list[Relation]) -> None:
        conn = self._get_conn()
        for rel in relations:
            conn.execute(
                """INSERT OR REPLACE INTO relations
                   (id, group_id, source_entity_id, target_entity_id, source_entity_name,
                    target_entity_name, name, fact, fact_embedding, episode_ids, attributes,
                    valid_at, invalid_at, expired_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (rel.id, rel.group_id, rel.source_entity_id,
                 rel.target_entity_id, rel.source_entity_name,
                 rel.target_entity_name, rel.name, rel.fact,
                 json.dumps(rel.fact_embedding or []),
                 json.dumps(rel.episode_ids), json.dumps(rel.attributes),
                 _normalize_ts(rel.valid_at), _normalize_ts(rel.invalid_at),
                 _normalize_ts(rel.expired_at),
                 _normalize_ts(rel.created_at)),
            )
        conn.commit()

    async def get_relation(self, relation_id: str) -> Relation | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM relations WHERE id = ?", (relation_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_relation(row)

    async def get_relations_between(
        self, source_id: str, target_id: str, group_id: str | None = None
    ) -> list[Relation]:
        """Get all relations between two entities (both directions)."""
        conn = self._get_conn()
        if group_id:
            rows = conn.execute(
                """SELECT * FROM relations
                   WHERE group_id = ? AND (
                     (source_entity_id = ? AND target_entity_id = ?) OR
                     (source_entity_id = ? AND target_entity_id = ?)
                   )""",
                (group_id, source_id, target_id, target_id, source_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM relations
                   WHERE (source_entity_id = ? AND target_entity_id = ?) OR
                         (source_entity_id = ? AND target_entity_id = ?)""",
                (source_id, target_id, target_id, source_id),
            ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    async def search_relations_by_text(
        self, group_id: str, query: str, filters: SearchFilters | None = None, limit: int = 20
    ) -> list[SearchResult]:
        """Text search over relation facts with temporal filtering."""
        conn = self._get_conn()
        query_lower = query.lower()

        sql = "SELECT * FROM relations WHERE group_id = ? AND LOWER(fact) LIKE ?"
        params: list[Any] = [group_id, f"%{query_lower}%"]

        sql, params = self._apply_temporal_filters(sql, params, filters)
        sql += " LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            rel = self._row_to_relation(row)
            # Score by substring match position (earlier = better)
            pos = rel.fact.lower().find(query_lower)
            score = 1.0 / (1.0 + pos) if pos >= 0 else 0.0
            results.append(SearchResult(relation=rel, score=score, source="text"))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def search_relations_by_embedding(
        self, group_id: str, embedding: list[float], filters: SearchFilters | None = None, limit: int = 20
    ) -> list[SearchResult]:
        """Vector search over relation fact embeddings with temporal filtering."""
        conn = self._get_conn()

        sql = "SELECT * FROM relations WHERE group_id = ?"
        params: list[Any] = [group_id]

        sql, params = self._apply_temporal_filters(sql, params, filters)
        rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            rel = self._row_to_relation(row)
            if rel.fact_embedding:
                sim = _cosine_similarity(embedding, rel.fact_embedding)
                results.append(SearchResult(relation=rel, score=sim, source="vector"))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def invalidate_relation(
        self, relation_id: str, invalid_at: str, expired_at: str | None = None
    ) -> None:
        """Mark a relation as invalidated (fact no longer true)."""
        conn = self._get_conn()
        if expired_at:
            conn.execute(
                "UPDATE relations SET invalid_at = ?, expired_at = ? WHERE id = ?",
                (_normalize_ts(invalid_at), _normalize_ts(expired_at), relation_id),
            )
        else:
            conn.execute(
                "UPDATE relations SET invalid_at = ? WHERE id = ?",
                (_normalize_ts(invalid_at), relation_id),
            )
        conn.commit()

    async def get_active_relations(
        self, group_id: str, filters: SearchFilters | None = None
    ) -> list[Relation]:
        """Get all currently active (not invalidated, not expired) relations."""
        conn = self._get_conn()
        sql = "SELECT * FROM relations WHERE group_id = ? AND invalid_at IS NULL AND expired_at IS NULL"
        params: list[Any] = [group_id]

        if filters and filters.valid_at_start:
            sql += " AND valid_at >= ?"
            params.append(_normalize_ts(filters.valid_at_start))
        if filters and filters.valid_at_end:
            sql += " AND valid_at <= ?"
            params.append(_normalize_ts(filters.valid_at_end))

        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_relation(r) for r in rows]

    # ------------------------------------------------------------------
    # Episodic links
    # ------------------------------------------------------------------

    async def save_episodic_links(self, links: list[EpisodicLink]) -> None:
        conn = self._get_conn()
        for link in links:
            conn.execute(
                """INSERT OR IGNORE INTO episodic_links
                   (id, episode_id, entity_id, group_id, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (link.id, link.episode_id, link.entity_id, link.group_id, link.created_at),
            )
        conn.commit()

    # ------------------------------------------------------------------
    # Temporal filter helper
    # ------------------------------------------------------------------

    def _apply_temporal_filters(
        self, sql: str, params: list[Any], filters: SearchFilters | None
    ) -> tuple[str, list[Any]]:
        """Apply temporal and partition filters to a SQL query."""
        if filters is None:
            # Default: exclude invalidated and expired
            sql += " AND invalid_at IS NULL AND expired_at IS NULL"
            return sql, params

        if not filters.include_invalidated:
            sql += " AND invalid_at IS NULL"
        if not filters.include_expired:
            sql += " AND expired_at IS NULL"
        if filters.valid_at_start:
            sql += " AND valid_at >= ?"
            params.append(_normalize_ts(filters.valid_at_start))
        if filters.valid_at_end:
            sql += " AND valid_at <= ?"
            params.append(_normalize_ts(filters.valid_at_end))
        if filters.relation_names:
            placeholders = ",".join("?" for _ in filters.relation_names)
            sql += f" AND name IN ({placeholders})"
            params.extend(filters.relation_names)
        if filters.entity_names:
            placeholders = ",".join("?" for _ in filters.entity_names)
            sql += f" AND (source_entity_name IN ({placeholders}) OR target_entity_name IN ({placeholders}))"
            params.extend(filters.entity_names)
            params.extend(filters.entity_names)

        return sql, params

    # ------------------------------------------------------------------
    # Row converters
    # ------------------------------------------------------------------

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"],
            group_id=row["group_id"],
            name=row["name"],
            content=row["content"],
            source=row["source"],
            episode_type=EpisodeType(row["episode_type"]),
            reference_time=row["reference_time"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]),
        )

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            group_id=row["group_id"],
            name=row["name"],
            entity_type=EntityType(row["entity_type"]),
            summary=row["summary"],
            name_embedding=json.loads(row["name_embedding"]),
            attributes=json.loads(row["attributes"]),
            episode_ids=json.loads(row["episode_ids"]),
            created_at=row["created_at"],
        )

    def _row_to_relation(self, row: sqlite3.Row) -> Relation:
        return Relation(
            id=row["id"],
            group_id=row["group_id"],
            source_entity_id=row["source_entity_id"],
            target_entity_id=row["target_entity_id"],
            source_entity_name=row["source_entity_name"],
            target_entity_name=row["target_entity_name"],
            name=row["name"],
            fact=row["fact"],
            fact_embedding=json.loads(row["fact_embedding"]),
            episode_ids=json.loads(row["episode_ids"]),
            attributes=json.loads(row["attributes"]),
            valid_at=row["valid_at"],
            invalid_at=row["invalid_at"],
            expired_at=row["expired_at"],
            created_at=row["created_at"],
        )
