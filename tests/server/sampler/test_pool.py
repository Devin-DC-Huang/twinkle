# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pool decisions against a controllable driver, without Ray or GPU side effects."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from twinkle.server.config.application_spec import SamplerArgs
from twinkle.server.sampler.controller import SamplerController


@pytest.fixture(scope='session', autouse=True)
def _ray_runtime():
    yield


class Driver:

    def __init__(self):
        self.started = []
        self.stopped = []
        self.calls = []
        self.finish = asyncio.Event()
        self.start_gate = asyncio.Event()
        self.start_gate.set()
        self.fail_start = False
        self.fail_close = False

    async def start(self, instance):
        self.started.append(instance.identity)
        await self.start_gate.wait()
        if self.fail_start:
            raise RuntimeError('allocation failed')
        instance.devices = [str(len(self.started))]

    async def invoke(self, instance, ticket, payload, queue=None):
        self.calls.append((instance.identity, payload))
        return {'status': 200, 'headers': [], 'body': b'{"status":"pending"}'}

    async def wait_operation(self, instance, ticket):
        await self.finish.wait()

    async def load_weights(self, instance, path):
        self.calls.append((instance.identity, {'base_path': path}))

    async def shutdown(self, instance):
        if self.fail_close:
            raise RuntimeError('death unconfirmed')
        self.stopped.append(instance.identity)

    async def recover_orphans(self):
        pass


def setup(minimum=2, maximum=4, target=1):
    args = SamplerArgs(
        model_id='mock',
        sampler_type='mock',
        device_group={
            'name': 'sampler',
            'ranks': 1,
            'device_type': 'cpu'
        },
        device_mesh={'dp_size': 1},
        min_replicas=minimum,
        max_replicas=maximum,
        target_ongoing_requests=target,
        idle_timeout_s=10)
    driver = Driver()
    clock = SimpleNamespace(now=0)
    pool = SamplerController(
        args, driver, {
            'startup_timeout_s': 2,
            'shutdown_timeout_s': 1,
            'sweep_interval_s': 1
        }, clock=lambda: clock.now)
    return pool, driver, clock


def request(endpoint='twinkle/sample', body=None):
    return {
        'endpoint': endpoint,
        'body': json.dumps(body or {}).encode(),
        'method': 'POST',
        'headers': [(b'x-request-id', b'test')],
        'query_string': b''
    }


async def settle_starts(pool):
    await asyncio.gather(*tuple(pool.lifecycle.startups.values()), return_exceptions=True)


async def settle_jobs(pool):
    await asyncio.gather(*tuple(pool.lifecycle.jobs), return_exceptions=True)


@pytest.mark.asyncio
async def test_minimum_expand_and_reclaim_only_idle_extras():
    pool, driver, clock = setup()
    await pool.initialize()
    assert len(driver.started) == 2
    try:
        for _ in range(8):
            await pool.request(request())
            await settle_starts(pool)
        assert len(driver.started) == 4 and pool._owned_count() == 4
        live = pool._ready_instances()
        assert all(instance.requests for instance in live)
        assert max(len(instance.requests) for instance in live) - min(len(instance.requests) for instance in live) <= 1
        clock.now = 30
        await pool.sweep()
        assert not driver.stopped
        driver.finish.set()
        await settle_jobs(pool)
        clock.now += 11
        await pool.sweep()
        await settle_jobs(pool)
        assert len(driver.stopped) == 2 and len(pool._ready_instances()) == 2
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
async def test_waiter_cancellation_preserves_generation_and_defers_control():
    pool, driver, _ = setup()
    await pool.initialize()
    await pool.request(request())
    control = asyncio.create_task(pool.request(request('twinkle/set_template', {'template_cls': 'Template'})))
    try:
        await asyncio.sleep(0)
        control.cancel()
        await asyncio.gather(control, return_exceptions=True)
        assert pool._guard._readers == 1
        assert sum(len(instance.requests) for instance in pool.instances.values()) == 1
        assert len(driver.calls) == 1
        driver.finish.set()
        await settle_jobs(pool)
        assert any(payload['endpoint'] == 'twinkle/set_template' for _, payload in driver.calls)
        assert pool._actions[0]['endpoint'] == 'twinkle/set_template'
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
async def test_new_instance_replays_controls_not_lora_weight_cache():
    pool, driver, _ = setup(minimum=1, target=1)
    driver.finish.set()
    await pool.initialize()
    try:
        await pool.request(request('twinkle/set_template', {'template_cls': 'Template'}))
        driver.finish.clear()
        await pool.request(request(body={'adapter_uri': 'checkpoint-A'}))
        await pool.request(request(body={'adapter_uri': 'checkpoint-B'}))
        driver.finish.set()
        await settle_starts(pool)
        await settle_jobs(pool)
        assert len(driver.started) == 2
        new = driver.started[-1]
        replay = [payload for identity, payload in driver.calls if identity == new]
        assert replay and all(payload['endpoint'] == 'twinkle/set_template' for payload in replay)
        assert all('adapter_uri' not in json.loads(payload['body']) for payload in pool._actions)
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
async def test_quarantine_counts_against_maximum_until_confirmed_death():
    pool, driver, _ = setup(minimum=1, maximum=1)
    await pool.initialize()
    instance = pool._ready_instances()[0]
    driver.fail_close = True
    await pool._close(instance)
    try:
        assert instance.state == 'QUARANTINED'
        for _ in range(3):
            await pool.sweep()
            await settle_jobs(pool)
        assert len(driver.started) == 1 and pool._owned_count() == 1
        driver.fail_close = False
        await pool.sweep()
        await settle_jobs(pool)
        await pool.sweep()
        await settle_starts(pool)
        assert len(driver.started) == 2 and len(pool._ready_instances()) == 1
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
async def test_startup_failure_rolls_back_loaded_minimum():
    pool, driver, _ = setup()
    original = driver.start

    async def fail_second(instance):
        if len(driver.started) == 1:
            driver.fail_start = True
        await original(instance)

    driver.start = fail_second
    with pytest.raises(RuntimeError, match='allocation'):
        await pool.initialize()
    assert not pool._ready and pool._owned_count() == 0


@pytest.mark.asyncio
async def test_lazy_first_requests_share_one_startup():
    pool, driver, _ = setup(minimum=0, maximum=1)
    await pool.initialize()
    driver.start_gate.clear()
    calls = [asyncio.create_task(pool.request(request())) for _ in range(8)]
    try:
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(driver.started) == 1
        driver.start_gate.set()
        await asyncio.wait_for(asyncio.gather(*calls), 2)
        assert len(driver.started) == 1
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['raise', 'http_error'])
async def test_partial_control_broadcast_fences_all_participants(fault):
    pool, driver, _ = setup()
    driver.finish.set()
    await pool.initialize()
    invoke = driver.invoke
    changed = []

    async def update(instance, ticket, payload, queue=None):
        changed.append(instance.identity)
        if len(changed) == 2:
            if fault == 'raise':
                raise ConnectionError('update ACK unknown')
            return {'status': 500, 'headers': [], 'body': b'error'}
        return await invoke(instance, ticket, payload, queue)

    driver.invoke = update
    try:
        if fault == 'raise':
            with pytest.raises(ConnectionError):
                await pool.request(request('twinkle/set_template'))
        else:
            reply = await pool.request(request('twinkle/set_template'))
            assert reply['status'] == 500
        await settle_jobs(pool)
        assert len(driver.stopped) == 2
        assert not pool._ready_instances() and not pool._actions
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_unknown_generation_exit_keeps_quarantined_capacity_reserved():
    pool, driver, clock = setup(minimum=1, maximum=1)
    await pool.initialize()
    driver.fail_close = True

    async def unknown(*args):
        raise TimeoutError('generation completion unknown')

    driver.wait_operation = unknown
    await pool.request(request())
    await settle_jobs(pool)
    instance = next(iter(pool.instances.values()))
    assert instance.state == 'QUARANTINED' and instance.requests
    clock.now = 100
    await pool.sweep()
    await settle_jobs(pool)
    assert len(driver.started) == 1 and instance.requests
    driver.fail_close = False
    await pool.sweep()
    await settle_jobs(pool)
    assert instance.state == 'STOPPED' and not instance.requests
    await pool.shutdown()


@pytest.mark.asyncio
async def test_training_and_sampling_use_the_same_lifecycle_owner():
    from tests.server.state.fake_backend import FakeBackend
    from tests.server.training.test_controller import setup as training_setup
    pool, driver, _ = setup()
    training, _, _, _ = await training_setup(backend=FakeBackend())
    try:
        assert type(training.lifecycle) is type(pool.lifecycle)
        assert training.lifecycle is not pool.lifecycle
    finally:
        driver.finish.set()
        await pool.shutdown()
        await training.shutdown()


@pytest.mark.asyncio
async def test_replaying_config_has_the_same_startup_deadline():
    pool, driver, _ = setup(minimum=0, maximum=1)
    await pool.initialize()
    pool._actions.append(request('twinkle/set_template'))
    pool.budgets['startup_timeout_s'] = 0.05
    instance = pool._new_instance()
    with pytest.raises(TimeoutError):
        await pool.lifecycle.startups[instance.identity]
    assert instance.state == 'STOPPED' and driver.stopped == [instance.identity]
    await pool.shutdown()


@pytest.mark.asyncio
async def test_shared_health_rule_tolerates_transient_probe_timeout():
    from twinkle.server.resources.instances import InstanceLifecycle
    clock = SimpleNamespace(now=0)
    lifecycle = InstanceLifecycle('health', clock=lambda: clock.now)

    async def unreachable():
        raise TimeoutError('control-plane probe timed out')

    async def healthy():
        return {'unhealthy': False}

    async def failed():
        return {'unhealthy': True}

    assert await lifecycle.health('instance', unreachable) is None
    clock.now = 29
    assert await lifecycle.health('instance', unreachable) is None
    clock.now = 30
    assert await lifecycle.health('instance', unreachable) is False
    assert await lifecycle.health('instance', healthy) is True
    assert not lifecycle.unreachable_since
    assert await lifecycle.health('instance', unreachable) is None
    assert await lifecycle.health('instance', failed) is False
