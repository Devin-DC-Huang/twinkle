import asyncio
from types import SimpleNamespace
import logging
import pytest
from twinkle.server.resources.instances import InstanceLifecycle


@pytest.mark.asyncio
async def test_spawn_logs_exception_with_traceback(caplog, monkeypatch):
    # The ``twinkle`` logger disables propagation, so attach caplog to the lifecycle logger directly.
    target = logging.getLogger('twinkle.server.resources.instances')
    monkeypatch.setattr(target, 'handlers', [*target.handlers, caplog.handler])
    lifecycle = InstanceLifecycle('Training')

    async def broken():
        raise RuntimeError('background sentinel')

    with caplog.at_level(logging.ERROR):
        lifecycle.spawn(broken())
        await lifecycle.join()
    records = [record for record in caplog.records if record.getMessage() == 'Training background job failed']
    assert records and records[0].exc_info[1].args == ('background sentinel',)
    assert not lifecycle.jobs




@pytest.mark.asyncio
async def test_concurrent_close_shares_one_stop_and_retries_quarantine():
    import asyncio
    lifecycle = InstanceLifecycle('test')
    entered, resume = asyncio.Event(), asyncio.Event()
    calls, events = [], []

    async def stop():
        calls.append('stop')
        entered.set()
        await resume.wait()
        if len(calls) == 1:
            raise RuntimeError('death unconfirmed')

    async def stopped():
        events.append('stopped')

    async def quarantined(error):
        events.append(str(error))

    first = asyncio.create_task(lifecycle.close('one', stop, 1, stopped, quarantined))
    await entered.wait()
    second = asyncio.create_task(lifecycle.close('one', stop, 1, stopped, quarantined))
    await asyncio.sleep(0)
    resume.set()
    await asyncio.gather(first, second)
    assert calls == ['stop'] and events == ['death unconfirmed']
    await lifecycle.close('one', stop, 1, stopped, quarantined)
    assert calls == ['stop', 'stop'] and events[-1] == 'stopped'
    await lifecycle.join()
    assert not lifecycle.jobs and not lifecycle.closing

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


@pytest.mark.asyncio
async def test_shutdown_actor_lookup_cannot_block_the_close_deadline(monkeypatch):
    import threading
    from unittest.mock import AsyncMock
    import ray
    from ray import serve
    from twinkle.server.resources.runtime import RayRuntime

    runtime = RayRuntime('test', allocation_timeout=1, shutdown_timeout=1)
    entered, resume, returned = threading.Event(), threading.Event(), threading.Event()

    def lookup(*args, **kwargs):
        entered.set()
        try:
            # A bounded fallback makes the broken implementation fail rather than hang pytest.
            if not resume.wait(2):
                raise TimeoutError('blocked actor class import')
            raise ValueError('actor already gone')
        finally:
            returned.set()

    monkeypatch.setattr(ray.util, 'list_named_actors', lambda **kwargs: [{'namespace': 'test', 'name': 'owned-worker'}])
    monkeypatch.setattr(ray, 'get_actor', lookup)
    monkeypatch.setattr(serve, 'delete', lambda name: None)
    runtime.pool.remove = AsyncMock()
    stopped, quarantined = AsyncMock(), AsyncMock()
    lifecycle = InstanceLifecycle('test')
    try:
        await lifecycle.close('owned', lambda: runtime.shutdown('owned', 'app'), .02, stopped, quarantined)
        assert entered.is_set()
        quarantined.assert_awaited_once()
        stopped.assert_not_awaited()
        assert not returned.is_set(), 'Synchronous actor lookup blocked the event loop past the close deadline'
    finally:
        resume.set()
        await asyncio.to_thread(returned.wait, 3)
        await lifecycle.join()
