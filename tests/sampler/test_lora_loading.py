# Copyright (c) ModelScope Contributors. All rights reserved.
"""Concurrent callers must share one vLLM adapter load, not duplicate IDs."""
import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine


@pytest.mark.asyncio
async def test_concurrent_lora_loads_share_request_and_retry_failure(monkeypatch, tmp_path):
    module = ModuleType('vllm.lora.request')
    module.LoRARequest = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', module)
    (tmp_path / 'adapter_config.json').write_text('{}')
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._lora_request_cache = {}
    engine._lora_load_tasks = {}
    engine._next_lora_id = 1
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    fail = False

    async def add(request):
        calls.append(request)
        entered.set()
        await release.wait()
        if fail:
            raise RuntimeError('load failed')

    engine.engine = SimpleNamespace(add_lora=add)
    jobs = [asyncio.create_task(engine._get_or_load_lora(str(tmp_path))) for _ in range(8)]
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.sleep(0.02)
    release.set()
    results = await asyncio.gather(*jobs)
    assert len(calls) == 1
    assert all(item is results[0] for item in results)
    assert await engine._get_or_load_lora(str(tmp_path)) is results[0]
    engine._lora_request_cache.clear()
    fail = True
    assert await engine._get_or_load_lora(str(tmp_path)) is None
    assert not engine._lora_load_tasks
    fail = False
    assert await engine._get_or_load_lora(str(tmp_path)) is not None
