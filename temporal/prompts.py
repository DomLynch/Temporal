"""
temporal/prompts.py — Extraction and resolution prompts.

Transplanted from Graphiti's prompts/ directory (1,293 LOC → ~350 LOC).
Preserved the core prompt IP that drives extraction quality.

Three prompt categories:
1. Entity extraction — message/text/json variants (episode-type-aware)
2. Relation extraction — fact triples with temporal fields
3. Resolution — entity dedup + relation duplicate/contradiction adjudication

All prompts return lists of {"role": str, "content": str} dicts
compatible with any OpenAI-compatible LLM client.
"""

from __future__ import annotations

import json
from typing import Any


def _to_json(obj: Any) -> str:
    """Format an object as readable JSON for prompts."""
    if isinstance(obj, list):
        return json.dumps(obj, indent=2, default=str)
    if isinstance(obj, dict):
        return json.dumps(obj, indent=2, default=str)
    return str(obj)


# ---------------------------------------------------------------------------
# Entity extraction prompts (episode-type-aware)
# ---------------------------------------------------------------------------

def extract_entities_message(context: dict[str, Any]) -> list[dict[str, str]]:
    """Extract entities from a conversational message."""
    return [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that extracts entity nodes from conversational messages. "
                "Extract and classify the speaker and other significant entities mentioned in the conversation."
            ),
        },
        {
            "role": "user",
            "content": f"""
<PREVIOUS MESSAGES>
{_to_json(context.get('previous_episodes', []))}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{context.get('episode_content', '')}
</CURRENT MESSAGE>

Extract all entity nodes mentioned in the CURRENT MESSAGE.

Rules:
1. Always extract the speaker (before the colon) as the first entity.
2. Extract significant entities, concepts, or actors explicitly or implicitly mentioned.
3. Do NOT extract entities only in PREVIOUS MESSAGES (context only).
4. Do NOT extract relationships, actions, or temporal information.
5. Use full names when available.
6. Resolve pronouns (he/she/they) to actual entity names.

Return JSON:
{{"entities": [{{"name": "Entity Name", "entity_type": "person|organization|location|concept|event|other"}}]}}
""",
        },
    ]


def extract_entities_text(context: dict[str, Any]) -> list[dict[str, str]]:
    """Extract entities from a text document."""
    return [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that extracts entity nodes from text. "
                "Extract and classify significant entities mentioned in the provided text."
            ),
        },
        {
            "role": "user",
            "content": f"""
<TEXT>
{context.get('episode_content', '')}
</TEXT>

Extract entities from the TEXT that are explicitly or implicitly mentioned.

Rules:
1. Extract significant entities, concepts, or actors.
2. Do NOT extract relationships or actions.
3. Do NOT extract dates or temporal information.
4. Use full names, avoid abbreviations.

Return JSON:
{{"entities": [{{"name": "Entity Name", "entity_type": "person|organization|location|concept|event|other"}}]}}
""",
        },
    ]


def extract_entities_json(context: dict[str, Any]) -> list[dict[str, str]]:
    """Extract entities from structured JSON data."""
    return [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that extracts entity nodes from JSON data. "
                "Extract and classify relevant entities from the provided JSON."
            ),
        },
        {
            "role": "user",
            "content": f"""
<SOURCE DESCRIPTION>
{context.get('source_description', 'Structured data')}
</SOURCE DESCRIPTION>

<JSON>
{context.get('episode_content', '')}
</JSON>

Extract relevant entities from the JSON.

Rules:
1. Extract entities from "name", "user", and similar identity fields.
2. Extract entities mentioned in other properties.
3. Do NOT extract date properties as entities.

Return JSON:
{{"entities": [{{"name": "Entity Name", "entity_type": "person|organization|location|concept|event|other"}}]}}
""",
        },
    ]


# Map episode type to extraction prompt
ENTITY_EXTRACTION_PROMPTS = {
    "message": extract_entities_message,
    "text": extract_entities_text,
    "json": extract_entities_json,
}


# ---------------------------------------------------------------------------
# Relation extraction prompt
# ---------------------------------------------------------------------------

def extract_relations(context: dict[str, Any]) -> list[dict[str, str]]:
    """Extract fact triples (relations) between entities with temporal fields."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert fact extractor. Extract fact triples from text. "
                "Include relevant date information. Treat CURRENT TIME as when the message was sent. "
                "All temporal information should be extracted relative to this time."
            ),
        },
        {
            "role": "user",
            "content": f"""
<PREVIOUS MESSAGES>
{_to_json(context.get('previous_episodes', []))}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{context.get('episode_content', '')}
</CURRENT MESSAGE>

<ENTITIES>
{_to_json(context.get('entities', []))}
</ENTITIES>

<REFERENCE TIME>
{context.get('reference_time', '')}
</REFERENCE TIME>

Extract all factual relationships between the ENTITIES based on the CURRENT MESSAGE.

Rules:
1. source_entity_name and target_entity_name MUST be names from the ENTITIES list.
2. Each fact must involve two DISTINCT entities.
3. No duplicate or semantically redundant facts.
4. The fact should closely paraphrase the source (not verbatim quote).
5. Use REFERENCE TIME to resolve relative temporal expressions ("last week", "yesterday").
6. Do NOT hallucinate temporal bounds from unrelated events.
7. Derive relation_name from the relationship predicate (e.g., "lives_in", "works_at", "founded").

Datetime rules:
- Use ISO 8601 with "Z" suffix (e.g., 2025-04-30T00:00:00Z).
- If ongoing (present tense), set valid_at to REFERENCE TIME.
- If a change/termination is expressed, set invalid_at to the relevant timestamp.
- Leave both null if no time is stated.

Return JSON:
{{"relations": [{{
    "source_entity_name": "...",
    "target_entity_name": "...",
    "relation_name": "...",
    "fact": "...",
    "valid_at": "ISO datetime or null",
    "invalid_at": "ISO datetime or null"
}}]}}
""",
        },
    ]


# ---------------------------------------------------------------------------
# Resolution prompts
# ---------------------------------------------------------------------------

def resolve_entity_dedup(context: dict[str, Any]) -> list[dict[str, str]]:
    """Determine if a new entity is a duplicate of an existing one."""
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that determines whether a NEW ENTITY "
                "is a duplicate of any EXISTING ENTITIES."
            ),
        },
        {
            "role": "user",
            "content": f"""
<CURRENT MESSAGE>
{context.get('episode_content', '')}
</CURRENT MESSAGE>

<NEW ENTITY>
{_to_json(context.get('new_entity', {}))}
</NEW ENTITY>

<EXISTING ENTITIES>
{_to_json(context.get('existing_entities', []))}
</EXISTING ENTITIES>

Determine if the NEW ENTITY is a duplicate of any EXISTING ENTITY.

Rules:
- Only consider duplicates if they refer to the SAME real-world object or concept.
- Semantic equivalence counts (e.g., "the CEO" and "John Smith" if context confirms).
- Do NOT mark related-but-distinct entities as duplicates.

Return JSON:
{{"is_duplicate": true/false, "duplicate_of": "name of existing entity or empty string", "best_name": "most complete name"}}
""",
        },
    ]


def resolve_relation_dedup(context: dict[str, Any]) -> list[dict[str, str]]:
    """Determine if a new relation duplicates or contradicts existing ones."""
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that de-duplicates facts and determines "
                "which existing facts are contradicted by a new fact."
            ),
        },
        {
            "role": "user",
            "content": f"""
<EXISTING FACTS>
{_to_json(context.get('existing_relations', []))}
</EXISTING FACTS>

<INVALIDATION CANDIDATES>
{_to_json(context.get('invalidation_candidates', []))}
</INVALIDATION CANDIDATES>

<NEW FACT>
{_to_json(context.get('new_relation', {}))}
</NEW FACT>

Tasks:
1. DUPLICATE DETECTION: If the NEW FACT represents identical information as any EXISTING FACT,
   return those indices in duplicate_indices.
2. CONTRADICTION DETECTION: Determine which facts the NEW FACT contradicts from either list.
   Return all contradicted indices in contradicted_indices.

Rules:
- Similar facts with key differences (especially numeric values) are NOT duplicates.
- A fact can be both a duplicate AND contradicted (e.g., same fact but new version supersedes).
- Indices are continuous across both lists (EXISTING FACTS start at 0, INVALIDATION CANDIDATES continue the numbering). Use the idx values shown next to each fact.

Return JSON:
{{"duplicate_indices": [int], "contradicted_indices": [int], "is_new": true/false}}
""",
        },
    ]
