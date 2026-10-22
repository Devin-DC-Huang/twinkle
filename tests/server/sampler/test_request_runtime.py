# Copyright (c) ModelScope Contributors. All rights reserved.
"""Actual completion handles outlive future timeout and streaming iterator closure."""
import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.server.utils.test_task_queue_mixin import _DummyQueue
from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.sampler.concurrency import SamplerConcurrency
from twinkle.server.task_queue.types import ExecutionScope, execution_scope


@pytest.mark.asyncio
async def test_worker_exports_actual_completion_handle_without_private_task_lookup():
    queue = _DummyQueue()
    queue._task_queue_config = TaskQueueConfig(execution_timeout=0.05)
    queue.enable_compute_worker()
    guard = SamplerConcurrency()
    entered, exit_gate = asyncio.Event(), asyncio.Event()
    scope = ExecutionScope()

    async def backend():
        entered.set()
        await exit_gate.wait()
        return 'late result'

    token = execution_scope.set(scope)
    try:
        ref = await queue.schedule_concurrent_task(lambda: guard.run(backend))
    finally:
        execution_scope.reset(token)
    try:
        await asyncio.wait_for(entered.wait(), 1)
        completion = asyncio.create_task(scope.wait())
        for _ in range(100):
            record = await queue.state.get_future(ref['request_id'])
            if record['status'] == 'failed':
                break
            await asyncio.sleep(0.01)
        assert record['status'] == 'failed'
        assert not completion.done() and guard._readers == 1
        assert scope.request_ids == {ref['request_id']} and len(scope.jobs) == 1
        exit_gate.set()
        await asyncio.wait_for(completion, 1)
        assert guard._readers == 0
    finally:
        exit_gate.set()
        await queue._compute_worker.stop()


def test_stream_iterator_close_joins_the_real_engine_loop_task():
    # Execute the repository method without constructing a GPU-backed vLLM engine.
    path = Path('src/twinkle/sampler/vllm_sampler/vllm_sampler.py')
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_iter_in_loop')
    namespace = {'asyncio': asyncio}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    loop = asyncio.new_event_loop()
    running, finalizing, release, closed = [threading.Event() for _ in range(4)]

    def run_loop():
        asyncio.set_event_loop(loop)
        running.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop)
    thread.start()
    running.wait(1)

    async def generation():
        try:
            yield 'first'
        finally:
            finalizing.set()
            await asyncio.to_thread(release.wait)

    iterator = namespace['_iter_in_loop'](SimpleNamespace(_async_loop=loop), generation())
    closer = None
    try:
        assert next(iterator) == 'first'
        assert finalizing.wait(1)

        def close():
            iterator.close()
            closed.set()

        closer = threading.Thread(target=close)
        closer.start()
        assert not closed.wait(0.05)
        release.set()
        assert closed.wait(2)
    finally:
        release.set()
        if closer:
            closer.join(2)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)
        loop.close()


@pytest.mark.asyncio
async def test_runtime_does_not_certify_an_unconfirmed_cancel_ack():
    from twinkle.server.sampler.runtime import SamplerRuntime
    from twinkle.server.task_queue.types import BackendBusyError
    guard = SamplerConcurrency()
    runtime = SamplerRuntime(SimpleNamespace(sampling_guard=guard))
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    runtime._operations['cancelled'] = {'task': task, 'scope': ExecutionScope()}
    guard.invalidate_weights(RuntimeError('cancel ACK preceded actual generation exit'))
    with pytest.raises(BackendBusyError):
        await runtime.wait_operation('cancelled')
    assert 'cancelled' in runtime._operations


@pytest.mark.asyncio
async def test_cold_lora_config_does_not_block_sampler_health(monkeypatch):
    import json
    import sys
    from twinkle.server.sampler.runtime import SamplerRuntime
    entered, release = threading.Event(), threading.Event()
    calls = []

    class ColdConfig:

        def __init__(self, **kwargs):
            entered.set()
            assert release.wait(2)

    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(LoraConfig=ColdConfig))

    async def call_backend(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def add_adapter(name, config):
        calls.append((name, config))

    owner = SimpleNamespace(
        model_id='mock',
        sampling_guard=SamplerConcurrency(),
        sampler=SimpleNamespace(add_adapter_to_sampler=add_adapter),
        call_backend=call_backend)
    runtime = SamplerRuntime(owner)
    payload = {
        'endpoint': 'twinkle/add_adapter_to_sampler',
        'method': 'POST',
        'headers': [(b'content-type', b'application/json'), (b'x-request-id', b'configuration')],
        'body': json.dumps({
            'adapter_name': 'a',
            'config': {
                'r': 4
            }
        }).encode()
    }
    invocation = asyncio.create_task(runtime.invoke('configuration', payload))
    try:
        began = await asyncio.to_thread(entered.wait, 5)
        assert began, invocation.result() if invocation.done() else 'Configuration route did not start'
        assert (await asyncio.wait_for(runtime.runtime_identity(), 0.1))['unhealthy'] is False
        assert not invocation.done() and not calls
        release.set()
        assert (await invocation)['status'] == 200
        await runtime.wait_operation('configuration')
        assert len(calls) == 1
    finally:
        release.set()
        await asyncio.gather(invocation, return_exceptions=True)


@pytest.mark.asyncio
async def test_managed_compute_http_cannot_bypass_pool_admission(monkeypatch):
    import httpx
    from fastapi import FastAPI
    from twinkle.server.exceptions import RequestRejectedError
    from twinkle.server.sampler import app as module

    def scaffold(title, register_routes, **kwargs):
        app = FastAPI()
        register_routes(app, lambda: SimpleNamespace())
        return app

    monkeypatch.setattr(module, 'build_deployment_app', scaffold)
    monkeypatch.setattr(module, 'bind_deployment', lambda app, *args, **kwargs: app)
    args = {
        'model_id': 'mock',
        'nproc_per_node': 1,
        'device_group': {
            'name': 'sampler',
            'ranks': 1,
            'device_type': 'cpu'
        },
        'device_mesh': {
            'dp_size': 1
        },
        'deploy_options': {},
        'sampler_type': 'mock'
    }
    managed = module.build_sampler_app(**args, sampling_context={'instance_id': 'owned'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=managed), base_url='http://test') as client:
        with pytest.raises(RequestRejectedError, match='stable sampler endpoint'):
            await client.post('/twinkle/create', json={})
    standalone = module.build_sampler_app(**args)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=standalone), base_url='http://test') as client:
        assert (await client.post('/twinkle/create', json={})).status_code == 200
