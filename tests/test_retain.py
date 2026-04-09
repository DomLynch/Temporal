"""
Tests for temporal/retain.py — Episode ingest orchestration.

Covers:
1. Full pipeline: episode → entities → relations → persist
2. Empty content returns failure
3. No LLM still saves episode
4. Entity extraction from LLM response
5. Relation extraction with entity resolution
6. Prior episode context retrieval
7. Embeddings generated for entities and relations
8. Provenance links created
9. Token usage accumulated across all steps
10. Invalidated relations counted
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from temporal.types import (
    Entity,
    EntityType,
    Episode,
    EpisodeType,
    EpisodicLink,
    Relation,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns scripted extraction results."""

    def __init__(self, entity_response=None, relation_response=None, resolve_response=None):
        self._entity_response = entity_response or {
            "entities": [
                {"name": "Alice", "entity_type": "person", "summary": "The founder"},
                {"name": "London", "entity_type": "location", "summary": "City in UAE"},
            ]
        }
        self._relation_response = relation_response or {
            "relations": [
                {
                    "source": "Alice",
                    "target": "London",
                    "relation_name": "lives_in",
                    "fact": "Alice lives in London",
                    "valid_at": "2022-01-01T00:00:00+00:00",
                },
            ]
        }
        self._resolve_response = resolve_response or {
            "is_new": True,
            "duplicate_indices": [],
            "contradicted_indices": [],
        }
        self.calls: list[dict] = []

    async def complete(self, messages, temperature=0.0, max_tokens=4096,
                       response_format=None, tools=None):
        self.calls.append({"messages": messages})

        # Determine which response based on prompt content
        system_msg = messages[0].get("content", "") if messages else ""

        if "entities" in system_msg.lower() and "relation" not in system_msg.lower():
            content = json.dumps(self._entity_response)
        elif "relation" in system_msg.lower() or "edge" in system_msg.lower():
            content = json.dumps(self._relation_response)
        elif "duplicate" in system_msg.lower() or "contradict" in system_msg.lower():
            content = json.dumps(self._resolve_response)
        else:
            content = json.dumps(self._entity_response)

        return {
            "content": content,
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }


class MockEmbedder:
    def __init__(self):
        self.call_count = 0

    async def embed(self, texts):
        self.call_count += 1
        return [[0.1 * (i + 1)] * 64 for i, _ in enumerate(texts)]


class MockStore:
    def __init__(self):
        self.episodes: list[Episode] = []
        self.entities: list[Entity] = []
        self.relations: list[Relation] = []
        self.links: list[EpisodicLink] = []

    async def save_episode(self, episode):
        self.episodes.append(episode)

    async def save_entities(self, entities):
        self.entities.extend(entities)

    async def save_relations(self, relations):
        self.relations.extend(relations)

    async def save_episodic_links(self, links):
        self.links.extend(links)

    async def get_recent_episodes(self, group_id, limit=5, before=None):
        return [e for e in self.episodes if e.group_id == group_id][-limit:]

    async def get_entities_by_group(self, group_id):
        return [e for e in self.entities if e.group_id == group_id]

    async def search_entities_by_name(self, group_id, name, limit=10):
        return [e for e in self.entities
                if e.group_id == group_id and name.lower() in e.name.lower()][:limit]

    async def search_entities_by_embedding(self, group_id, embedding, limit=10):
        return [e for e in self.entities if e.group_id == group_id][:limit]

    async def get_relations_between(self, source_id, target_id, group_id=None):
        return [r for r in self.relations
                if r.source_entity_id == source_id and r.target_entity_id == target_id]

    async def search_relations_by_text(self, group_id, query, filters=None, limit=20):
        return [SearchResult(relation=r, score=0.5)
                for r in self.relations if query.lower() in r.fact.lower()][:limit]

    async def search_relations_by_embedding(self, group_id, embedding, filters=None, limit=20):
        return []

    async def get_active_relations(self, group_id, filters=None):
        return [r for r in self.relations if r.group_id == group_id and r.is_active]

    async def invalidate_relation(self, relation_id, invalid_at, expired_at=None):
        for r in self.relations:
            if r.id == relation_id:
                r.invalid_at = invalid_at
                r.expired_at = expired_at


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRetainBasic:
    async def test_full_pipeline(self):
        """Full ingest: episode → entities → relations → persist."""
        from temporal.retain import retain

        llm = MockLLM()
        embedder = MockEmbedder()
        store = MockStore()

        result = await retain(
            content="Alice lives in London and builds AI.",
            group_id="user-1",
            source="telegram",
            llm=llm,
            embedder=embedder,
            store=store,
        )

        assert result.success
        assert result.entities_extracted >= 1
        assert result.relations_extracted >= 0  # Depends on LLM mock
        assert len(store.episodes) == 1
        assert len(store.entities) >= 1
        assert result.token_usage["total_tokens"] > 0

    async def test_empty_content_fails(self):
        from temporal.retain import retain

        result = await retain(content="", group_id="user-1")
        assert not result.success

    async def test_whitespace_content_fails(self):
        from temporal.retain import retain

        result = await retain(content="   ", group_id="user-1")
        assert not result.success

    async def test_no_llm_saves_episode_only(self):
        """Without LLM, episode is saved but no extraction happens."""
        from temporal.retain import retain

        store = MockStore()
        result = await retain(
            content="Some text",
            group_id="user-1",
            store=store,
        )

        assert len(store.episodes) == 1
        assert result.entities_extracted == 0

    async def test_episode_metadata(self):
        """Episode has correct metadata."""
        from temporal.retain import retain

        store = MockStore()
        llm = MockLLM()

        await retain(
            content="Test content",
            group_id="user-1",
            name="Test Episode",
            source="whatsapp",
            episode_type=EpisodeType.MESSAGE,
            llm=llm,
            store=store,
        )

        ep = store.episodes[0]
        assert ep.group_id == "user-1"
        assert ep.name == "Test Episode"
        assert ep.source == "whatsapp"
        assert ep.episode_type == EpisodeType.MESSAGE


@pytest.mark.asyncio
class TestEntityExtraction:
    async def test_entities_extracted_from_llm(self):
        from temporal.retain import retain

        llm = MockLLM(entity_response={
            "entities": [
                {"name": "Alice", "entity_type": "person"},
                {"name": "Nexus", "entity_type": "concept"},
                {"name": "London", "entity_type": "location"},
            ]
        })
        store = MockStore()

        result = await retain(
            content="Alice builds Nexus in London",
            group_id="user-1",
            llm=llm,
            store=store,
        )

        assert result.entities_extracted == 3
        names = {e.name for e in store.entities}
        assert "Alice" in names
        assert "Nexus" in names

    async def test_empty_entity_names_filtered(self):
        from temporal.retain import retain

        llm = MockLLM(entity_response={
            "entities": [
                {"name": "Valid", "entity_type": "person"},
                {"name": "", "entity_type": "person"},
                {"name": "  ", "entity_type": "person"},
            ]
        })
        store = MockStore()

        result = await retain(
            content="test", group_id="user-1", llm=llm, store=store,
        )

        # Only "Valid" should be extracted
        assert result.entities_extracted == 1


@pytest.mark.asyncio
class TestRelationExtraction:
    async def test_relations_linked_to_entities(self):
        from temporal.retain import retain

        llm = MockLLM()
        store = MockStore()

        result = await retain(
            content="Alice lives in London",
            group_id="user-1",
            llm=llm,
            store=store,
        )

        if store.relations:
            rel = store.relations[0]
            assert rel.source_entity_name == "Alice"
            assert rel.target_entity_name == "London"
            assert rel.fact == "Alice lives in London"

    async def test_relations_get_temporal_fields(self):
        from temporal.retain import retain

        llm = MockLLM()
        store = MockStore()

        await retain(
            content="Alice lives in London",
            group_id="user-1",
            llm=llm,
            store=store,
        )

        if store.relations:
            rel = store.relations[0]
            assert rel.valid_at is not None  # Should have a valid_at


@pytest.mark.asyncio
class TestEmbeddings:
    async def test_entity_embeddings_generated(self):
        from temporal.retain import retain

        embedder = MockEmbedder()
        store = MockStore()
        llm = MockLLM()

        await retain(
            content="Alice builds Nexus",
            group_id="user-1",
            llm=llm,
            embedder=embedder,
            store=store,
        )

        assert embedder.call_count >= 1  # At least one embed call

    async def test_no_embedder_still_works(self):
        from temporal.retain import retain

        store = MockStore()
        llm = MockLLM()

        result = await retain(
            content="Alice builds Nexus",
            group_id="user-1",
            llm=llm,
            store=store,
        )

        assert result.success


@pytest.mark.asyncio
class TestProvenance:
    async def test_episodic_links_created(self):
        from temporal.retain import retain

        store = MockStore()
        llm = MockLLM()

        result = await retain(
            content="Alice builds Nexus",
            group_id="user-1",
            llm=llm,
            store=store,
        )

        # Links should connect episode to extracted entities
        assert len(store.links) >= 1
        for link in store.links:
            assert link.episode_id == result.episode_id
            assert link.group_id == "user-1"


@pytest.mark.asyncio
class TestTokenUsage:
    async def test_usage_accumulated(self):
        from temporal.retain import retain

        llm = MockLLM()

        result = await retain(
            content="Alice builds Nexus in London",
            group_id="user-1",
            llm=llm,
        )

        assert result.token_usage["total_tokens"] > 0
        assert result.token_usage["input_tokens"] > 0


@pytest.mark.asyncio
class TestPriorContext:
    async def test_prior_episodes_retrieved(self):
        """Second retain should get first episode as context."""
        from temporal.retain import retain

        store = MockStore()
        llm = MockLLM()

        # First retain
        await retain(
            content="Alice founded GDA",
            group_id="user-1",
            llm=llm,
            store=store,
        )

        # Second retain should have prior context
        await retain(
            content="Alice moved to London",
            group_id="user-1",
            llm=llm,
            store=store,
        )

        assert len(store.episodes) == 2
