# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backported sampler concurrency must retain the deployed synchronous wire API."""
import asyncio
import threading
import time

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from twinkle.data_format import SampledSequence, SampleResponse
from twinkle.server.sampler.app import SamplerManagement
from twinkle.server.sampler.concurrency import SamplerConcurrency
from twinkle.server.sampler.tinker_handlers import _register_tinker_sampler_routes
from twinkle.server.sampler.twinkle_handlers import _register_twinkle_sampler_routes
from twinkle.server.utils.task_queue import TaskQueueConfig


class State:
    def __init__(self):
        self.records = {}

    async def store_future_status(self, request_id, status, model_id, **values):
        self.records[request_id] = dict(status=status, model_id=model_id, **values)

    async def get_future(self, request_id):
        return self.records.get(request_id)

    async def get_sampling_session(self, _session):
        return {'model_path': None}


class BlockingSampler:
    def __init__(self):
        self.release = threading.Event()
        self.release.set()
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.weights = None
        self.weight_events = []
        self.fail_load = False

    def sample(self, inputs, sampling_params, **kwargs):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if not self.release.wait(5):
                raise TimeoutError('test did not release backend')
            return [SampleResponse(sequences=[SampledSequence(
                tokens=[entry['input_ids'][0]], stop_reason='length', logprobs=[[(entry['input_ids'][0], -0.1)]],
                decoded=str(self.weights)) for _ in range(sampling_params.num_samples)]) for entry in inputs]
        finally:
            with self.lock:
                self.active -= 1

    def load_full_weights_from_path(self, path):
        self.weights = path
        self.weight_events.append(path)
        if self.fail_load:
            raise RuntimeError('load failed after a partial write')

    def set_template(self, *args, **kwargs):
        self.weight_events.append('template')


@pytest_asyncio.fixture
async def make_client():
    services = []
    clients = []

    def make(*, timeout=2, limit=32):
        service = SamplerManagement.__new__(SamplerManagement)
        service.model_id = 'legacy-test'
        service.sampler_type = 'mock'
        service.sampler = BlockingSampler()
        service.state = State()
        service.sampling_guard = SamplerConcurrency()
        service._stream_jobs = set()
        service._init_task_queue(TaskQueueConfig(execution_timeout=timeout, concurrent_task_limit=limit), 'Sampler')

        async def start(_request):
            service.sampling_guard.check_available()
            return 'EMPTY_TOKEN'

        service._on_request_start = start
        app = FastAPI()
        _register_twinkle_sampler_routes(app, lambda: service)
        _register_tinker_sampler_routes(app, lambda: service)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://legacy')
        services.append(service)
        clients.append(client)
        return service, client

    yield make
    for service in services:
        service.sampler.release.set()
    for service in services:
        await service.shutdown()
    for client in clients:
        await client.aclose()


async def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, 'test backend did not reach the expected state'
        await asyncio.sleep(0.005)


def sample_body(value=7, **extra):
    return {'inputs': [{'input_ids': [value]}], 'sampling_params': {'max_tokens': 1},
            'adapter_name': '', 'num_samples': 1, **extra}


@pytest.mark.asyncio
async def test_old_twinkle_wire_overlaps_and_create_stays_responsive(make_client):
    service, client = make_client()
    service.sampler.release.clear()
    pending = [asyncio.create_task(client.post('/twinkle/sample', json=sample_body(value))) for value in range(8)]
    await wait_for(lambda: service.sampler.active == 8)
    health = await asyncio.wait_for(client.post('/twinkle/create', json={}), timeout=0.5)
    assert health.status_code == 200
    assert health.json() == {'status': 'ok'}
    service.sampler.release.set()
    responses = await asyncio.gather(*pending)
    assert service.sampler.peak == 8
    for value, response in enumerate(responses):
        assert response.status_code == 200
        assert 'request_id' not in response.json()
        assert response.json()['samples'][0]['sequences'][0]['tokens'] == [value]


@pytest.mark.asyncio
async def test_old_top_level_num_samples_and_flat_template_kwargs(make_client):
    service, client = make_client()
    response = await client.post('/twinkle/sample', json=sample_body(num_samples=3))
    assert response.status_code == 200
    assert len(response.json()['samples'][0]['sequences']) == 3
    response = await client.post('/twinkle/set_template', json={
        'template_cls': 'Template', 'adapter_name': '', 'model_id': 'legacy-test'})
    assert response.status_code == 200
    assert service.sampler.weight_events == ['template']


@pytest.mark.asyncio
async def test_tinker_submit_returns_before_generation_finishes(make_client):
    from tinker import types
    service, client = make_client()
    service.sampler.release.clear()
    body = types.SampleRequest(prompt=types.ModelInput.from_ints([9]), num_samples=2,
                               sampling_params=types.SamplingParams(max_tokens=1, temperature=0))
    response = await asyncio.wait_for(client.post('/tinker/asample', json=body.model_dump(mode='json')), timeout=0.5)
    assert response.status_code == 200
    request_id = response.json()['request_id']
    await wait_for(lambda: service.sampler.active == 1)
    assert service.state.records[request_id]['status'] == 'running'
    service.sampler.release.set()
    await wait_for(lambda: service.state.records[request_id]['status'] == 'completed')
    result = service.state.records[request_id]['result']
    assert len(result.sequences) == 2
    assert result.sequences[0].tokens == [9]


@pytest.mark.asyncio
async def test_capacity_rejects_without_creating_a_future_and_recovers(make_client):
    service, client = make_client(limit=1)
    service.sampler.release.clear()
    first = asyncio.create_task(client.post('/twinkle/sample', json=sample_body()))
    await wait_for(lambda: service.sampler.active == 1)
    before = set(service.state.records)
    rejected = await client.post('/twinkle/sample', json=sample_body())
    assert rejected.status_code == 429
    assert rejected.headers['retry-after'] == '1'
    assert set(service.state.records) == before
    service.sampler.release.set()
    assert (await first).status_code == 200
    await wait_for(lambda: service._compute_worker._detached_reserved == 0)
    assert (await client.post('/twinkle/sample', json=sample_body())).status_code == 200


@pytest.mark.asyncio
async def test_timeout_is_terminal_but_does_not_free_live_gpu_capacity(make_client):
    service, client = make_client(timeout=0.05, limit=1)
    service.sampler.release.clear()
    response = await asyncio.wait_for(client.post('/twinkle/sample', json=sample_body()), timeout=0.5)
    assert response.status_code == 504
    assert service.sampler.active == 1
    assert service._compute_worker._detached_reserved == 1
    request_id = next(iter(service.state.records))
    assert service.state.records[request_id]['status'] == 'failed'
    assert (await client.post('/twinkle/sample', json=sample_body())).status_code == 429
    service.sampler.release.set()
    await wait_for(lambda: service._compute_worker._detached_reserved == 0)
    assert service.state.records[request_id]['status'] == 'failed'
    assert (await client.post('/twinkle/sample', json=sample_body())).status_code == 200


@pytest.mark.asyncio
async def test_template_mutation_waits_for_existing_generations(make_client):
    service, client = make_client()
    service.sampler.release.clear()
    generating = asyncio.create_task(client.post('/twinkle/sample', json=sample_body()))
    await wait_for(lambda: service.sampler.active == 1)
    mutation = asyncio.create_task(client.post('/twinkle/set_template', json={
        'template_cls': 'Template', 'adapter_name': ''}))
    await asyncio.sleep(0.03)
    assert not mutation.done()
    assert service.sampler.weight_events == []
    service.sampler.release.set()
    assert (await generating).status_code == 200
    assert (await mutation).status_code == 200


@pytest.mark.asyncio
async def test_full_checkpoint_load_and_generation_stay_one_transaction(make_client, tmp_path):
    service, client = make_client()
    # Use the actual filesystem classification, but no fake model weights.
    checkpoints = [str(tmp_path / 'A'), str(tmp_path / 'B')]
    import twinkle.server.checkpoint
    from unittest.mock import patch
    class Resolver:
        def parse_adapter_uri(self, value):
            return '', value
    with patch.object(twinkle.server.checkpoint, 'create_checkpoint_manager', return_value=Resolver()):
        responses = await asyncio.gather(*[
            client.post('/twinkle/sample', json=sample_body(adapter_uri=path)) for path in checkpoints])
    assert [response.json()['samples'][0]['sequences'][0]['decoded'] for response in responses] == checkpoints
    assert service.sampler.weight_events == checkpoints


@pytest.mark.asyncio
async def test_shutdown_marks_live_future_failed_then_waits_for_backend_exit(make_client):
    service, client = make_client()
    service.sampler.release.clear()
    waiting = asyncio.create_task(client.post('/twinkle/sample', json=sample_body()))
    await wait_for(lambda: service.sampler.active == 1)
    closing = asyncio.create_task(service.shutdown())
    await wait_for(lambda: next(iter(service.state.records.values()))['status'] == 'failed')
    assert not closing.done()
    service.sampler.release.set()
    await closing
    assert (await waiting).status_code == 500


@pytest.mark.asyncio
async def test_failed_weight_write_quarantines_the_sampler(make_client):
    service, client = make_client()
    service.sampler.fail_load = True
    with pytest.raises(RuntimeError, match='partial write'):
        await service.run_sample(lambda: service.load_sample_weights('/failed-checkpoint'), exclusive=True)
    response = await client.post('/twinkle/sample', json=sample_body())
    assert response.status_code == 503
    assert service.sampler.active == 0
