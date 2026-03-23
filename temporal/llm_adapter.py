"""
temporal/llm_adapter.py — Bridge between LLMClient and Temporal's prompt system.

Graphiti uses a complex `generate_response()` pattern with:
- prompt_name lookup
- response_model (Pydantic) for structured output
- group_id threading
- model_size selection

Temporal simplifies this to:
- Prompt functions return message lists
- LLM client parses JSON from content
- No Pydantic dependency for response parsing

This adapter handles:
1. Calling the LLM with prompt-generated messages
2. Parsing JSON responses (with <think> tag stripping for Qwen models)
3. Extracting structured data from LLM output
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from temporal.interfaces import LLMClient

_log = logging.getLogger("temporal.llm_adapter")

# Regex to strip <think>...</think> blocks (Qwen models)
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


async def llm_extract(
    llm: LLMClient,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Call LLM with messages and parse JSON response.

    Args:
        llm: LLM client implementing the LLMClient protocol.
        messages: Chat messages from a prompt function.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.

    Returns:
        (parsed_data, usage) where parsed_data is the JSON response
        and usage is token counts.
    """
    response = await llm.complete(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    content = response.get("content", "")
    usage = response.get("usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    # Strip thinking tags if present (Qwen models)
    if "<think>" in content:
        content = _THINK_RE.sub("", content).strip()

    parsed = _parse_json_response(content)
    return parsed, usage


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse LLM response as JSON, with fallback extraction.

    Handles:
    - Clean JSON: {"key": "value"}
    - JSON in markdown: ```json\n{...}\n```
    - JSON embedded in text: some text {"key": "value"} more text
    - Empty/malformed: returns empty dict
    """
    if not content or not content.strip():
        return {}

    content = content.strip()

    # Try direct parse
    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"items": result}
        # Non-dict/list (float, int, str, etc.) — wrap it
        return {"value": result}
    except json.JSONDecodeError:
        pass

    # Try stripping markdown code blocks
    md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if md_match:
        try:
            result = json.loads(md_match.group(1).strip())
            return result if isinstance(result, dict) else {"value": result}
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return result if isinstance(result, dict) else {"value": result}
        except json.JSONDecodeError:
            pass

    _log.warning("Could not parse LLM response as JSON: %s", content[:200])
    return {}


def accumulate_usage(
    total: dict[str, int],
    new: dict[str, int],
) -> dict[str, int]:
    """Accumulate token usage across multiple LLM calls."""
    return {
        "input_tokens": total.get("input_tokens", 0) + new.get("input_tokens", 0),
        "output_tokens": total.get("output_tokens", 0) + new.get("output_tokens", 0),
        "total_tokens": total.get("total_tokens", 0) + new.get("total_tokens", 0),
    }
