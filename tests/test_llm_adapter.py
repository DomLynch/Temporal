"""
Tests for temporal/llm_adapter.py — LLM bridge.

Covers:
1. JSON parsing from clean LLM output
2. JSON parsing from markdown code blocks
3. JSON parsing from text with embedded JSON
4. <think> tag stripping (Qwen models)
5. Empty/malformed response handling
6. Usage accumulation
7. Full llm_extract flow with mock client
"""

import pytest

from temporal.llm_adapter import (
    _parse_json_response,
    accumulate_usage,
    llm_extract,
)


class TestParseJsonResponse:
    def test_clean_json(self):
        result = _parse_json_response('{"key": "value", "count": 3}')
        assert result == {"key": "value", "count": 3}

    def test_json_in_markdown(self):
        result = _parse_json_response('```json\n{"entities": ["Alice"]}\n```')
        assert result == {"entities": ["Alice"]}

    def test_json_in_markdown_no_lang(self):
        result = _parse_json_response('```\n{"data": true}\n```')
        assert result == {"data": True}

    def test_json_embedded_in_text(self):
        result = _parse_json_response('Here are the results: {"facts": ["A", "B"]} end.')
        assert result == {"facts": ["A", "B"]}

    def test_empty_string(self):
        assert _parse_json_response("") == {}

    def test_none_handling(self):
        # None shouldn't be passed but handle gracefully
        assert _parse_json_response(None) == {}

    def test_whitespace_only(self):
        assert _parse_json_response("   \n  ") == {}

    def test_completely_invalid(self):
        assert _parse_json_response("This is not JSON at all") == {}

    def test_nested_json(self):
        result = _parse_json_response('{"entities": [{"name": "Dom", "type": "person"}]}')
        assert result["entities"][0]["name"] == "Dom"

    def test_think_tags_in_content(self):
        """Qwen models wrap output in <think>...</think> tags."""
        content = '<think>Let me analyze this carefully...</think>{"entities": [{"name": "Alice"}]}'
        result = _parse_json_response(content)
        # Note: _parse_json_response doesn't strip think tags — that's done in llm_extract
        # But the JSON extractor should find the JSON after the think block
        assert "entities" in result


class TestAccumulateUsage:
    def test_basic_accumulation(self):
        total = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        new = {"input_tokens": 200, "output_tokens": 30, "total_tokens": 230}
        result = accumulate_usage(total, new)
        assert result == {"input_tokens": 300, "output_tokens": 80, "total_tokens": 380}

    def test_empty_initial(self):
        total = {}
        new = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        result = accumulate_usage(total, new)
        assert result == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

    def test_empty_new(self):
        total = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        result = accumulate_usage(total, {})
        assert result == total


@pytest.mark.asyncio
class TestLLMExtract:
    async def test_basic_extract(self):
        class MockLLM:
            async def complete(self, messages, temperature=0.0, max_tokens=4096,
                             response_format=None, tools=None):
                return {
                    "content": '{"entities": [{"name": "Alice"}]}',
                    "usage": {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
                }

        data, usage = await llm_extract(MockLLM(), [{"role": "user", "content": "test"}])
        assert data["entities"][0]["name"] == "Alice"
        assert usage["total_tokens"] == 80

    async def test_think_tag_stripping(self):
        class ThinkingLLM:
            async def complete(self, messages, temperature=0.0, max_tokens=4096,
                             response_format=None, tools=None):
                return {
                    "content": '<think>Analyzing...</think>{"result": "clean"}',
                    "usage": {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
                }

        data, usage = await llm_extract(ThinkingLLM(), [{"role": "user", "content": "test"}])
        assert data["result"] == "clean"

    async def test_empty_response(self):
        class EmptyLLM:
            async def complete(self, messages, temperature=0.0, max_tokens=4096,
                             response_format=None, tools=None):
                return {"content": "", "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10}}

        data, usage = await llm_extract(EmptyLLM(), [{"role": "user", "content": "test"}])
        assert data == {}
        assert usage["total_tokens"] == 10

    async def test_json_format_requested(self):
        """Verify response_format is passed to the LLM."""
        class TrackingLLM:
            def __init__(self):
                self.last_format = None

            async def complete(self, messages, temperature=0.0, max_tokens=4096,
                             response_format=None, tools=None):
                self.last_format = response_format
                return {"content": "{}", "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}

        llm = TrackingLLM()
        await llm_extract(llm, [{"role": "user", "content": "test"}])
        assert llm.last_format == {"type": "json_object"}
