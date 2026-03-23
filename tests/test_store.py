"""
Tests for temporal/store.py — SQLite temporal store.

Covers GPT's required invariants:
1. Episode CRUD + recent episode retrieval
2. Entity CRUD + name search + embedding search
3. Relation CRUD + endpoint lookup
4. Temporal filtering (active vs invalidated vs expired)
5. Invalidation (setting invalid_at/expired_at)
6. Partition isolation via group_id
7. Episodic links (provenance)
8. Text search with temporal filtering
9. Vector search with temporal filtering
10. Round-trip persistence
"""

import pytest

from temporal.types import (
    Entity,
    EntityType,
    Episode,
    EpisodeType,
    EpisodicLink,
    Relation,
    SearchFilters,
)


@pytest.fixture
def store(tmp_path):
    from temporal.store import SQLiteTemporalStore
    db_path = tmp_path / "test_temporal.db"
    s = SQLiteTemporalStore(db_path=db_path)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEpisodes:
    async def test_save_and_get(self, store):
        ep = Episode(id="ep1", group_id="brain", content="Hello", source="telegram")
        await store.save_episode(ep)

        loaded = await store.get_episode("ep1")
        assert loaded is not None
        assert loaded.content == "Hello"
        assert loaded.source == "telegram"

    async def test_get_missing_returns_none(self, store):
        assert await store.get_episode("nonexistent") is None

    async def test_recent_episodes(self, store):
        for i in range(5):
            await store.save_episode(Episode(
                id=f"ep{i}", group_id="brain",
                reference_time=f"2026-03-{20+i}T10:00:00",
            ))

        recent = await store.get_recent_episodes("brain", limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].id == "ep4"

    async def test_recent_episodes_with_before(self, store):
        for i in range(5):
            await store.save_episode(Episode(
                id=f"ep{i}", group_id="brain",
                reference_time=f"2026-03-{20+i}T10:00:00",
            ))

        recent = await store.get_recent_episodes("brain", limit=3, before="2026-03-23T10:00:00")
        assert len(recent) == 3
        assert recent[0].id == "ep2"  # March 22


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEntities:
    async def test_save_and_get(self, store):
        ent = Entity(id="ent1", group_id="brain", name="Dominic", entity_type=EntityType.PERSON)
        await store.save_entity(ent)

        loaded = await store.get_entity("ent1")
        assert loaded is not None
        assert loaded.name == "Dominic"
        assert loaded.entity_type == EntityType.PERSON

    async def test_get_entities_by_group(self, store):
        await store.save_entity(Entity(id="e1", group_id="brain", name="Dom"))
        await store.save_entity(Entity(id="e2", group_id="brain", name="Dubai"))
        await store.save_entity(Entity(id="e3", group_id="other", name="Other"))

        brain_entities = await store.get_entities_by_group("brain")
        assert len(brain_entities) == 2

    async def test_search_by_name(self, store):
        await store.save_entity(Entity(id="e1", group_id="brain", name="Dominic Lynch"))
        await store.save_entity(Entity(id="e2", group_id="brain", name="Dubai"))

        results = await store.search_entities_by_name("brain", "dominic")
        assert len(results) == 1
        assert results[0].name == "Dominic Lynch"

    async def test_search_by_embedding(self, store):
        await store.save_entity(Entity(id="e1", group_id="brain", name="Dominic",
                                       name_embedding=[1.0, 0.0, 0.0]))
        await store.save_entity(Entity(id="e2", group_id="brain", name="Dubai",
                                       name_embedding=[0.0, 1.0, 0.0]))

        results = await store.search_entities_by_embedding("brain", [0.9, 0.1, 0.0])
        assert len(results) == 2
        assert results[0].name == "Dominic"  # Closer to query

    async def test_save_entities_bulk(self, store):
        entities = [
            Entity(id=f"e{i}", group_id="brain", name=f"Entity {i}")
            for i in range(10)
        ]
        await store.save_entities(entities)

        loaded = await store.get_entities_by_group("brain")
        assert len(loaded) == 10


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRelations:
    async def test_save_and_get(self, store):
        rel = Relation(
            id="r1", group_id="brain",
            source_entity_id="e1", target_entity_id="e2",
            source_entity_name="Dominic", target_entity_name="Dubai",
            name="lives_in", fact="Dominic lives in Dubai",
            valid_at="2022-01-01T00:00:00+00:00",
        )
        await store.save_relation(rel)

        loaded = await store.get_relation("r1")
        assert loaded is not None
        assert loaded.fact == "Dominic lives in Dubai"
        assert loaded.valid_at == "2022-01-01T00:00:00+00:00"
        assert loaded.is_active is True

    async def test_get_relations_between(self, store):
        await store.save_relation(Relation(
            id="r1", group_id="brain",
            source_entity_id="e1", target_entity_id="e2",
            fact="Dominic lives in Dubai",
        ))
        await store.save_relation(Relation(
            id="r2", group_id="brain",
            source_entity_id="e1", target_entity_id="e2",
            fact="Dominic works in Dubai",
        ))
        await store.save_relation(Relation(
            id="r3", group_id="brain",
            source_entity_id="e1", target_entity_id="e3",
            fact="Dominic knows Bob",
        ))

        between = await store.get_relations_between("e1", "e2", group_id="brain")
        assert len(between) == 2

    async def test_get_relations_between_bidirectional(self, store):
        """Should find relations regardless of direction."""
        await store.save_relation(Relation(
            id="r1", group_id="brain",
            source_entity_id="e1", target_entity_id="e2",
            fact="A relates to B",
        ))

        # Query in reverse direction
        between = await store.get_relations_between("e2", "e1", group_id="brain")
        assert len(between) == 1


# ---------------------------------------------------------------------------
# Temporal filtering — the core value
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTemporalFiltering:
    async def test_active_relations_excludes_invalidated(self, store):
        await store.save_relation(Relation(
            id="r1", group_id="brain", fact="Active fact",
            valid_at="2026-01-01T00:00:00",
        ))
        await store.save_relation(Relation(
            id="r2", group_id="brain", fact="Invalidated fact",
            valid_at="2025-01-01T00:00:00",
            invalid_at="2026-01-01T00:00:00",
        ))

        active = await store.get_active_relations("brain")
        assert len(active) == 1
        assert active[0].id == "r1"

    async def test_active_relations_excludes_expired(self, store):
        await store.save_relation(Relation(
            id="r1", group_id="brain", fact="Active",
        ))
        await store.save_relation(Relation(
            id="r2", group_id="brain", fact="Expired",
            expired_at="2026-01-01T00:00:00",
        ))

        active = await store.get_active_relations("brain")
        assert len(active) == 1
        assert active[0].id == "r1"

    async def test_invalidate_relation(self, store):
        await store.save_relation(Relation(
            id="r1", group_id="brain", fact="Will be invalidated",
            valid_at="2025-01-01T00:00:00",
        ))

        await store.invalidate_relation("r1", invalid_at="2026-03-01T00:00:00")

        loaded = await store.get_relation("r1")
        assert loaded.invalid_at == "2026-03-01T00:00:00"
        assert loaded.is_active is False

    async def test_invalidate_with_expiry(self, store):
        await store.save_relation(Relation(
            id="r1", group_id="brain", fact="Will be expired",
        ))

        await store.invalidate_relation(
            "r1",
            invalid_at="2026-03-01T00:00:00",
            expired_at="2026-03-01T00:00:01",
        )

        loaded = await store.get_relation("r1")
        assert loaded.invalid_at == "2026-03-01T00:00:00"
        assert loaded.expired_at == "2026-03-01T00:00:01"

    async def test_search_with_temporal_window(self, store):
        await store.save_relation(Relation(
            id="r1", group_id="brain", fact="Old fact about Dubai",
            valid_at="2020-01-01T00:00:00",
        ))
        await store.save_relation(Relation(
            id="r2", group_id="brain", fact="New fact about Dubai",
            valid_at="2026-01-01T00:00:00",
        ))

        filters = SearchFilters(valid_at_start="2025-01-01T00:00:00")
        results = await store.search_relations_by_text("brain", "dubai", filters=filters)
        assert len(results) == 1
        assert results[0].relation.id == "r2"

    async def test_search_includes_invalidated_when_requested(self, store):
        await store.save_relation(Relation(
            id="r1", group_id="brain", fact="Active fact",
        ))
        await store.save_relation(Relation(
            id="r2", group_id="brain", fact="Invalidated fact",
            invalid_at="2026-01-01T00:00:00",
        ))

        # Without flag: only active
        results = await store.search_relations_by_text("brain", "fact")
        assert len(results) == 1

        # With flag: include invalidated
        filters = SearchFilters(include_invalidated=True)
        results = await store.search_relations_by_text("brain", "fact", filters=filters)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Partition isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPartitionIsolation:
    async def test_entities_isolated_by_group(self, store):
        await store.save_entity(Entity(id="e1", group_id="brain", name="Dom"))
        await store.save_entity(Entity(id="e2", group_id="personal", name="Dom"))

        brain = await store.get_entities_by_group("brain")
        personal = await store.get_entities_by_group("personal")
        assert len(brain) == 1
        assert len(personal) == 1

    async def test_relations_isolated_by_group(self, store):
        await store.save_relation(Relation(id="r1", group_id="brain", fact="Brain fact"))
        await store.save_relation(Relation(id="r2", group_id="personal", fact="Personal fact"))

        brain = await store.get_active_relations("brain")
        personal = await store.get_active_relations("personal")
        assert len(brain) == 1
        assert len(personal) == 1

    async def test_search_isolated_by_group(self, store):
        await store.save_relation(Relation(id="r1", group_id="brain", fact="Dominic in brain"))
        await store.save_relation(Relation(id="r2", group_id="other", fact="Dominic in other"))

        results = await store.search_relations_by_text("brain", "dominic")
        assert len(results) == 1
        assert results[0].relation.group_id == "brain"


# ---------------------------------------------------------------------------
# Episodic links
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEpisodicLinks:
    async def test_save_links(self, store):
        links = [
            EpisodicLink(episode_id="ep1", entity_id="e1", group_id="brain"),
            EpisodicLink(episode_id="ep1", entity_id="e2", group_id="brain"),
        ]
        await store.save_episodic_links(links)
        # No error = success (links are write-only for now)


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestVectorSearch:
    async def test_search_by_embedding(self, store):
        await store.save_relation(Relation(
            id="r1", group_id="brain", fact="Dominic lives in Dubai",
            fact_embedding=[1.0, 0.0, 0.0],
        ))
        await store.save_relation(Relation(
            id="r2", group_id="brain", fact="Brain uses Qwen model",
            fact_embedding=[0.0, 1.0, 0.0],
        ))

        results = await store.search_relations_by_embedding(
            "brain", [0.9, 0.1, 0.0]
        )
        assert len(results) == 2
        assert results[0].relation.id == "r1"  # Closer to query
        assert results[0].source == "vector"

    async def test_vector_search_respects_temporal_filters(self, store):
        await store.save_relation(Relation(
            id="r1", group_id="brain", fact="Active",
            fact_embedding=[1.0, 0.0], valid_at="2026-01-01T00:00:00",
        ))
        await store.save_relation(Relation(
            id="r2", group_id="brain", fact="Invalidated",
            fact_embedding=[0.9, 0.1],
            invalid_at="2026-01-01T00:00:00",
        ))

        results = await store.search_relations_by_embedding("brain", [1.0, 0.0])
        assert len(results) == 1
        assert results[0].relation.id == "r1"


# ---------------------------------------------------------------------------
# Round-trip persistence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRoundTrip:
    async def test_full_lifecycle(self, store):
        """Episode → Entity → Relation → Invalidate → Search."""
        # Create episode
        ep = Episode(id="ep1", group_id="brain", content="Dom moved to Dubai in 2022")
        await store.save_episode(ep)

        # Create entities
        dom = Entity(id="e1", group_id="brain", name="Dominic", entity_type=EntityType.PERSON)
        dubai = Entity(id="e2", group_id="brain", name="Dubai", entity_type=EntityType.LOCATION)
        await store.save_entities([dom, dubai])

        # Create relation
        rel = Relation(
            id="r1", group_id="brain",
            source_entity_id="e1", target_entity_id="e2",
            source_entity_name="Dominic", target_entity_name="Dubai",
            name="lives_in", fact="Dominic lives in Dubai",
            valid_at="2022-01-01T00:00:00",
            episode_ids=["ep1"],
        )
        await store.save_relation(rel)

        # Create provenance link
        await store.save_episodic_links([
            EpisodicLink(episode_id="ep1", entity_id="e1", group_id="brain"),
            EpisodicLink(episode_id="ep1", entity_id="e2", group_id="brain"),
        ])

        # Verify active
        active = await store.get_active_relations("brain")
        assert len(active) == 1

        # New episode contradicts
        ep2 = Episode(id="ep2", group_id="brain", content="Dom moved to London")
        await store.save_episode(ep2)

        # Invalidate old relation
        await store.invalidate_relation("r1", invalid_at="2026-03-01T00:00:00")

        # Add new relation
        rel2 = Relation(
            id="r2", group_id="brain",
            source_entity_id="e1", target_entity_id="e3",
            source_entity_name="Dominic", target_entity_name="London",
            name="lives_in", fact="Dominic lives in London",
            valid_at="2026-03-01T00:00:00",
            episode_ids=["ep2"],
        )
        london = Entity(id="e3", group_id="brain", name="London", entity_type=EntityType.LOCATION)
        await store.save_entity(london)
        await store.save_relation(rel2)

        # Only new fact is active
        active = await store.get_active_relations("brain")
        assert len(active) == 1
        assert active[0].fact == "Dominic lives in London"

        # Old fact still retrievable with flag
        filters = SearchFilters(include_invalidated=True)
        all_rels = await store.search_relations_by_text("brain", "dominic", filters=filters)
        assert len(all_rels) == 2  # Both old and new
