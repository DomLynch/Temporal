"""
Tests for temporal/types.py — Core data types.

Covers:
1. Episode creation and defaults
2. Entity creation and types
3. Relation creation with temporal fields
4. Relation.is_active property
5. SearchFilters defaults
6. SearchResults convenience properties
7. RetainResult and ResolveResult
8. EpisodicLink provenance
9. ID generation uniqueness
10. Enum values
"""

from datetime import datetime, timezone

import pytest

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
    _new_id,
    _now_iso,
)


class TestHelpers:
    def test_new_id_is_16_chars(self):
        assert len(_new_id()) == 16

    def test_new_id_is_unique(self):
        ids = {_new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_now_iso_contains_timezone(self):
        ts = _now_iso()
        assert "+" in ts or "Z" in ts


class TestEpisode:
    def test_create_default(self):
        ep = Episode()
        assert ep.id
        assert ep.episode_type == EpisodeType.MESSAGE
        assert ep.group_id == ""
        assert ep.content == ""

    def test_create_with_values(self):
        ep = Episode(
            group_id="brain",
            name="Chat with Dom",
            content="Hey, what's up?",
            source="telegram",
            episode_type=EpisodeType.MESSAGE,
        )
        assert ep.group_id == "brain"
        assert ep.content == "Hey, what's up?"
        assert ep.source == "telegram"

    def test_episode_types(self):
        assert EpisodeType.MESSAGE.value == "message"
        assert EpisodeType.TEXT.value == "text"
        assert EpisodeType.JSON.value == "json"


class TestEntity:
    def test_create_default(self):
        ent = Entity()
        assert ent.id
        assert ent.entity_type == EntityType.OTHER
        assert ent.name == ""
        assert ent.episode_ids == []

    def test_create_person(self):
        ent = Entity(
            name="Dominic",
            entity_type=EntityType.PERSON,
            group_id="brain",
        )
        assert ent.name == "Dominic"
        assert ent.entity_type == EntityType.PERSON

    def test_entity_types(self):
        assert EntityType.PERSON.value == "person"
        assert EntityType.ORGANIZATION.value == "organization"
        assert EntityType.LOCATION.value == "location"
        assert EntityType.CONCEPT.value == "concept"

    def test_entity_with_embedding(self):
        ent = Entity(name="Test", name_embedding=[0.1, 0.2, 0.3])
        assert len(ent.name_embedding) == 3


class TestRelation:
    def test_create_default(self):
        rel = Relation()
        assert rel.id
        assert rel.valid_at is None
        assert rel.invalid_at is None
        assert rel.expired_at is None
        assert rel.is_active is True

    def test_active_relation(self):
        rel = Relation(
            source_entity_name="Dominic",
            target_entity_name="Dubai",
            name="lives_in",
            fact="Dominic lives in Dubai",
            valid_at="2022-01-01T00:00:00+00:00",
        )
        assert rel.is_active is True

    def test_invalidated_relation(self):
        rel = Relation(
            fact="Dominic lives in London",
            valid_at="2018-01-01T00:00:00+00:00",
            invalid_at="2022-01-01T00:00:00+00:00",
        )
        assert rel.is_active is False

    def test_expired_relation(self):
        rel = Relation(
            fact="Old fact",
            expired_at="2025-01-01T00:00:00+00:00",
        )
        assert rel.is_active is False

    def test_relation_with_episodes(self):
        rel = Relation(
            fact="Test fact",
            episode_ids=["ep_1", "ep_2"],
        )
        assert len(rel.episode_ids) == 2

    def test_relation_with_embedding(self):
        rel = Relation(fact="Test", fact_embedding=[0.1, 0.2])
        assert len(rel.fact_embedding) == 2


class TestEpisodicLink:
    def test_create_link(self):
        link = EpisodicLink(
            episode_id="ep_1",
            entity_id="ent_1",
            group_id="brain",
        )
        assert link.episode_id == "ep_1"
        assert link.entity_id == "ent_1"
        assert link.id  # auto-generated


class TestSearchFilters:
    def test_defaults(self):
        f = SearchFilters()
        assert f.group_ids is None
        assert f.include_invalidated is False
        assert f.include_expired is False
        assert f.limit == 20

    def test_temporal_filters(self):
        f = SearchFilters(
            valid_at_start="2025-01-01T00:00:00+00:00",
            valid_at_end="2026-01-01T00:00:00+00:00",
            include_invalidated=True,
        )
        assert f.valid_at_start is not None
        assert f.include_invalidated is True

    def test_partition_filter(self):
        f = SearchFilters(group_ids=["brain", "personal"])
        assert len(f.group_ids) == 2


class TestSearchResults:
    def test_empty_results(self):
        sr = SearchResults()
        assert sr.results == []
        assert sr.relations == []
        assert sr.total_found == 0

    def test_relations_property(self):
        rel1 = Relation(fact="Fact 1")
        rel2 = Relation(fact="Fact 2")
        sr = SearchResults(
            results=[
                SearchResult(relation=rel1, score=0.9),
                SearchResult(relation=rel2, score=0.7),
            ],
            total_found=2,
        )
        assert len(sr.relations) == 2
        assert sr.relations[0].fact == "Fact 1"


class TestRetainResult:
    def test_defaults(self):
        r = RetainResult()
        assert r.success is True
        assert r.entities_extracted == 0
        assert r.relations_invalidated == 0

    def test_with_counts(self):
        r = RetainResult(
            entities_extracted=5,
            entities_resolved=2,
            relations_extracted=8,
            relations_invalidated=1,
        )
        assert r.entities_extracted == 5
        assert r.relations_invalidated == 1


class TestResolveResult:
    def test_new_verdict(self):
        r = ResolveResult(verdict=ResolutionVerdict.NEW)
        assert r.verdict == ResolutionVerdict.NEW
        assert r.merged is False

    def test_duplicate_verdict(self):
        r = ResolveResult(
            verdict=ResolutionVerdict.DUPLICATE,
            canonical_id="existing_123",
            merged=True,
        )
        assert r.merged is True

    def test_contradicts_verdict(self):
        r = ResolveResult(
            verdict=ResolutionVerdict.CONTRADICTS,
            invalidated_ids=["old_1", "old_2"],
        )
        assert len(r.invalidated_ids) == 2

    def test_resolution_verdicts(self):
        assert ResolutionVerdict.NEW.value == "new"
        assert ResolutionVerdict.DUPLICATE.value == "duplicate"
        assert ResolutionVerdict.CONTRADICTS.value == "contradicts"
        assert ResolutionVerdict.UPDATE.value == "update"
