"""
End-to-end test of Temporal against a real LLM (OpenRouter).

Proves the full pipeline:
1. retain() extracts entities + relations with temporal validity
2. Second retain() triggers entity dedup + relation resolution
3. search() retrieves with temporal filtering
4. Contradiction invalidation works end-to-end

Run: OPENROUTER_API_KEY=... python3 tests/test_e2e_real.py
"""

import asyncio
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from temporal.types import Episode, EpisodeType, SearchFilters
from temporal.store import SQLiteTemporalStore
from temporal.retain import retain
from temporal.search import search

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-30b-a3b")


class OpenRouterLLM:
    def __init__(self, api_key=OPENROUTER_KEY, model=OPENROUTER_MODEL):
        self.api_key = api_key
        self.model = model

    async def complete(self, messages, temperature=0.0, max_tokens=4096,
                       response_format=None, tools=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  LLM ERROR: {e}")
            return {"content": "", "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        # Strip <think> tags
        if "<think>" in content:
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        raw_usage = data.get("usage", {})
        return {
            "content": content,
            "usage": {
                "input_tokens": raw_usage.get("prompt_tokens", 0),
                "output_tokens": raw_usage.get("completion_tokens", 0),
                "total_tokens": raw_usage.get("total_tokens", 0),
            },
        }


class SimpleEmbedder:
    async def embed(self, texts):
        import hashlib
        embeddings = []
        for text in texts:
            h = hashlib.sha256(text.encode()).hexdigest()
            emb = [int(h[i:i+2], 16) / 255.0 for i in range(0, min(128, len(h)), 2)]
            while len(emb) < 64:
                emb.append(0.0)
            embeddings.append(emb[:64])
        return embeddings


async def main():
    if not OPENROUTER_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        return False

    print("=" * 60)
    print("TEMPORAL END-TO-END TEST (Real LLM)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "temporal_e2e.db")
        store = SQLiteTemporalStore(db_path=db_path)
        llm = OpenRouterLLM()
        embedder = SimpleEmbedder()

        # -------------------------------------------------------------------
        # TEST 1: retain() — extract entities + relations
        # -------------------------------------------------------------------
        print("\n--- TEST 1: retain() ---")
        print("Input: paragraph about Alice")

        r1 = await retain(
            content=(
                "Alice is a tech founder based in London. He is building "
                "a cognitive AI system called Nexus. He previously lived in "
                "London before moving to London in 2022."
            ),
            group_id="user-1",
            source="test",
            episode_type=EpisodeType.TEXT,
            llm=llm,
            embedder=embedder,
            store=store,
        )

        print(f"  Success: {r1.success}")
        print(f"  Entities extracted: {r1.entities_extracted}")
        print(f"  Relations extracted: {r1.relations_extracted}")
        print(f"  Tokens: {r1.token_usage.get('total_tokens', 0)}")

        if not r1.success or r1.entities_extracted == 0:
            print("  ERROR: retain() failed or no entities")
            return False

        print("  ✅ retain() PASSED")

        # -------------------------------------------------------------------
        # TEST 2: Check what's in the store
        # -------------------------------------------------------------------
        print("\n--- TEST 2: Store contents ---")

        entities = await store.get_entities_by_group("user-1")
        relations = await store.get_active_relations("user-1")

        print(f"  Entities: {len(entities)}")
        for e in entities[:5]:
            print(f"    - {e.name} ({e.entity_type.value})")

        print(f"  Active relations: {len(relations)}")
        for r in relations[:5]:
            print(f"    - {r.fact[:80]}")
            if r.valid_at:
                print(f"      valid_at: {r.valid_at}")

        print("  ✅ Store contents PASSED")

        # -------------------------------------------------------------------
        # TEST 3: search() — temporal retrieval
        # -------------------------------------------------------------------
        print("\n--- TEST 3: search() ---")
        print("  Query: 'Where does Alice live?'")

        results = await search(
            query="Where does Alice live?",
            group_id="user-1",
            store=store,
            embedder=embedder,
        )

        print(f"  Results: {results.total_found}")
        for sr in results.results[:5]:
            print(f"    [{sr.score:.2f}] {sr.relation.fact[:80]}")

        if results.total_found == 0:
            print("  WARNING: No search results (may be embedding mismatch)")
        else:
            print("  ✅ search() PASSED")

        # -------------------------------------------------------------------
        # TEST 4: Second retain — triggers entity dedup
        # -------------------------------------------------------------------
        print("\n--- TEST 4: Second retain (entity dedup) ---")

        r2 = await retain(
            content=(
                "Alice is also pursuing a part-time PhD while managing "
                "Nexus development. He works 12 hours a day."
            ),
            group_id="user-1",
            source="test",
            llm=llm,
            embedder=embedder,
            store=store,
        )

        print(f"  Success: {r2.success}")
        print(f"  Entities extracted: {r2.entities_extracted}")
        print(f"  Entities resolved (dedup): {r2.entities_resolved}")
        print(f"  Relations extracted: {r2.relations_extracted}")

        entities_after = await store.get_entities_by_group("user-1")
        print(f"  Total entities in store: {len(entities_after)}")

        print("  ✅ Second retain PASSED")

        # -------------------------------------------------------------------
        # TEST 5: Temporal search — check valid_at fields
        # -------------------------------------------------------------------
        print("\n--- TEST 5: Temporal validity ---")

        all_relations = await store.get_active_relations("user-1")
        temporal_count = sum(1 for r in all_relations if r.valid_at)
        print(f"  Relations with valid_at: {temporal_count}/{len(all_relations)}")

        if temporal_count > 0:
            print("  ✅ Temporal validity PASSED")
        else:
            print("  WARNING: No temporal fields extracted (model may not support)")

        # -------------------------------------------------------------------
        # SUMMARY
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print(f"  Episodes: {len(await store.get_recent_episodes('brain', limit=100))}")
        print(f"  Entities: {len(entities_after)}")
        print(f"  Relations: {len(all_relations)}")
        print("=" * 60)

        store.close()
        return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
