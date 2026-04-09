"""
Tests for temporal/prompts.py — Extraction and resolution prompts.

Covers:
1. Entity extraction prompts (message/text/json variants)
2. Relation extraction prompt
3. Entity dedup resolution prompt
4. Relation dedup resolution prompt
5. Episode-type-aware prompt selection
6. Prompt format (list of role/content dicts)
"""

import pytest

from temporal.prompts import (
    ENTITY_EXTRACTION_PROMPTS,
    extract_entities_message,
    extract_entities_text,
    extract_entities_json,
    extract_relations,
    resolve_entity_dedup,
    resolve_relation_dedup,
)


class TestEntityExtractionPrompts:
    def test_message_prompt_structure(self):
        msgs = extract_entities_message({
            "previous_episodes": [{"content": "Previous chat"}],
            "episode_content": "Alice: I'm building Nexus in London",
        })
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "CURRENT MESSAGE" in msgs[1]["content"]
        assert "Nexus in London" in msgs[1]["content"]

    def test_text_prompt_structure(self):
        msgs = extract_entities_text({
            "episode_content": "Alice is the founder of Global Digital Assets.",
        })
        assert len(msgs) == 2
        assert "TEXT" in msgs[1]["content"]

    def test_json_prompt_structure(self):
        msgs = extract_entities_json({
            "episode_content": '{"name": "Alice", "role": "founder"}',
            "source_description": "User profile",
        })
        assert len(msgs) == 2
        assert "JSON" in msgs[1]["content"]
        assert "User profile" in msgs[1]["content"]

    def test_episode_type_map(self):
        assert "message" in ENTITY_EXTRACTION_PROMPTS
        assert "text" in ENTITY_EXTRACTION_PROMPTS
        assert "json" in ENTITY_EXTRACTION_PROMPTS
        assert ENTITY_EXTRACTION_PROMPTS["message"] is extract_entities_message
        assert ENTITY_EXTRACTION_PROMPTS["text"] is extract_entities_text
        assert ENTITY_EXTRACTION_PROMPTS["json"] is extract_entities_json

    def test_message_prompt_includes_previous_episodes(self):
        msgs = extract_entities_message({
            "previous_episodes": [{"content": "Earlier context"}],
            "episode_content": "Current msg",
        })
        assert "Earlier context" in msgs[1]["content"]

    def test_empty_context_handled(self):
        msgs = extract_entities_message({})
        assert len(msgs) == 2
        # Should not crash on missing keys


class TestRelationExtractionPrompt:
    def test_structure(self):
        msgs = extract_relations({
            "previous_episodes": [],
            "episode_content": "Alice founded Nexus in London",
            "entities": [{"name": "Alice"}, {"name": "Nexus"}, {"name": "London"}],
            "reference_time": "2026-03-22T10:00:00Z",
        })
        assert len(msgs) == 2
        assert "ENTITIES" in msgs[1]["content"]
        assert "REFERENCE TIME" in msgs[1]["content"]
        assert "Alice" in msgs[1]["content"]

    def test_includes_temporal_rules(self):
        msgs = extract_relations({"entities": [], "episode_content": "test"})
        content = msgs[1]["content"]
        assert "valid_at" in content
        assert "invalid_at" in content
        assert "ISO 8601" in content


class TestResolutionPrompts:
    def test_entity_dedup_structure(self):
        msgs = resolve_entity_dedup({
            "episode_content": "Alice mentioned Nexus",
            "new_entity": {"name": "Alice", "entity_type": "person"},
            "existing_entities": [{"name": "Alice Chen", "entity_type": "person"}],
        })
        assert len(msgs) == 2
        assert "NEW ENTITY" in msgs[1]["content"]
        assert "EXISTING ENTITIES" in msgs[1]["content"]

    def test_relation_dedup_structure(self):
        msgs = resolve_relation_dedup({
            "existing_relations": [{"fact": "Alice lives in London", "idx": 0}],
            "invalidation_candidates": [],
            "new_relation": {"fact": "Alice lives in London"},
        })
        assert len(msgs) == 2
        assert "EXISTING FACTS" in msgs[1]["content"]
        assert "NEW FACT" in msgs[1]["content"]
        assert "CONTRADICTION" in msgs[1]["content"].upper()

    def test_relation_dedup_includes_invalidation_candidates(self):
        msgs = resolve_relation_dedup({
            "existing_relations": [],
            "invalidation_candidates": [{"fact": "Old fact", "idx": 0}],
            "new_relation": {"fact": "New fact"},
        })
        assert "INVALIDATION CANDIDATES" in msgs[1]["content"]
