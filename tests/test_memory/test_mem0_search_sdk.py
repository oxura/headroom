"""Mem0 2.x retrieval against local Qdrant and locally cached embeddings."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from headroom.memory.backends.direct_mem0 import DirectMem0Adapter
from headroom.memory.backends.direct_mem0 import Mem0Config as DirectConfig
from headroom.memory.backends.mem0 import Mem0Backend
from headroom.memory.backends.mem0 import Mem0Config as BackendConfig
from headroom.memory.ports import VectorFilter

pytestmark = pytest.mark.slow

QUERY = "The Orion project listens on port 44123."


@pytest.fixture(scope="module")
def mem0_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    root = tmp_path_factory.mktemp("mem0-search")
    with pytest.MonkeyPatch.context() as env:
        env.setenv("MEM0_DIR", str(root / "mem0"))
        env.setenv("MEM0_TELEMETRY", "false")
        env.setenv("HF_HUB_OFFLINE", "1")
        sdk = pytest.importorskip("mem0")
        pytest.importorskip("sentence_transformers")
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            model = snapshot_download(
                "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
            )
        except LocalEntryNotFoundError:
            pytest.skip("Requires locally cached all-MiniLM-L6-v2 embeddings")

        client = sdk.Memory.from_config(
            {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "mem0_search",
                        "path": str(root / "qdrant"),
                        "embedding_model_dims": 384,
                        "on_disk": False,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": model,
                        "embedding_dims": 384,
                        "model_kwargs": {"device": "cpu", "local_files_only": True},
                    },
                },
                "llm": {
                    "provider": "openai",
                    "config": {"api_key": "unused-infer-false", "model": "gpt-4o-mini"},
                },
                "history_db_path": str(root / "history.db"),
            }
        )
        try:
            for user, text in [
                ("user-a", QUERY),
                ("user-a", "The Orion project stores data in PostgreSQL."),
                ("user-b", QUERY),
            ]:
                client.add(text, user_id=user, infer=False)
            yield client
        finally:
            client.close()
            client.vector_store.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter", "user_id", "filter_user"),
    [
        pytest.param("direct", "user-a", None, id="direct"),
        pytest.param("backend", "user-a", None, id="backend-explicit-scope"),
        pytest.param("backend", None, "user-a", id="backend-filter-scope"),
        pytest.param("backend", "user-a", "user-b", id="explicit-scope-precedence"),
    ],
)
async def test_search_returns_only_the_best_memory_for_the_requested_user(
    mem0_client: Any, adapter: str, user_id: str | None, filter_user: str | None
) -> None:
    if adapter == "direct":
        direct = DirectMem0Adapter(DirectConfig(enable_graph=False))
        direct._mem0_client = mem0_client
        direct._initialized = True
        results = await direct.search_memories(QUERY, user_id="user-a", top_k=1)
    else:
        backend = Mem0Backend(BackendConfig(enable_graph=False))
        backend._client = mem0_client
        backend._initialized = True
        scope = VectorFilter(user_id=filter_user) if filter_user is not None else None
        results = await backend.search_memories(QUERY, user_id=user_id, filter=scope, limit=1)

    assert [(row.memory.content, row.memory.user_id) for row in results] == [(QUERY, "user-a")]
