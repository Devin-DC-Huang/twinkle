# Copyright (c) ModelScope Contributors. All rights reserved.
"""Resource RPC deadlines must preserve ownership of late placement groups."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from twinkle.server.resources.allocation import DevicePool


@pytest.mark.asyncio
@pytest.mark.parametrize('blocked', ['nodes', 'available_resources', 'placement_group'])
async def test_cancelled_allocation_keeps_late_creation_owned(monkeypatch, blocked):
    import importlib
    import ray
    placement = importlib.import_module('ray.util.placement_group')
    entered, resume, returned = threading.Event(), threading.Event(), threading.Event()
    created, removed = [], []

    async def ready():
        return None

    pg = SimpleNamespace(id=SimpleNamespace(hex=lambda: 'pg-id'), ready=ready)

    def call(name, result):
        if name == blocked:
            entered.set()
            try:
                if not resume.wait(2):
                    raise TimeoutError('blocked allocation RPC')
            finally:
                returned.set()
        return result

    def create(*args, **kwargs):
        result = call('placement_group', pg)
        created.append(result)
        return result

    def get(name):
        if not created:
            raise ValueError('not created yet')
        return pg

    monkeypatch.setattr(ray, 'nodes', lambda: call('nodes', [{'Alive': True, 'Resources': {'CPU': 2}}]))
    monkeypatch.setattr(ray, 'available_resources', lambda: call('available_resources', {'CPU': 2}))
    monkeypatch.setattr(placement, 'placement_group', create)
    monkeypatch.setattr(placement, 'get_placement_group', get)
    monkeypatch.setattr(placement, 'remove_placement_group', lambda group: removed.append(group))
    monkeypatch.setattr(placement, 'placement_group_table', lambda group: {'state': 'REMOVED'})
    pool = DevicePool('test', allocation_timeout=.02, shutdown_timeout=.02)
    operation = asyncio.create_task(pool.allocate(1, 'cpu', 'epoch', 'instance'))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        with pytest.raises(TimeoutError):
            await operation
        assert not returned.is_set(), 'Allocation RPC blocked its deadline'
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(pool.remove(pool.name('epoch', 'instance')), .02)
        assert not created, 'The blocked RPC should still be pending'
    finally:
        resume.set()
        await asyncio.gather(operation, return_exceptions=True)
        await asyncio.to_thread(returned.wait, 3)
        await asyncio.wait_for(pool.remove(pool.name('epoch', 'instance')), 1)
    assert created and removed and all(group is pg for group in removed)


@pytest.mark.asyncio
@pytest.mark.parametrize('blocked', ['get_placement_group', 'remove_placement_group', 'placement_group_table'])
async def test_removal_rpc_cannot_block_quarantine_deadline(monkeypatch, blocked):
    import importlib
    placement = importlib.import_module('ray.util.placement_group')
    entered, resume, returned = threading.Event(), threading.Event(), threading.Event()
    pg = object()

    def call(name, result):
        if name == blocked:
            entered.set()
            try:
                if not resume.wait(2):
                    raise TimeoutError('blocked placement group RPC')
            finally:
                returned.set()
        return result

    monkeypatch.setattr(placement, 'get_placement_group', lambda name: call('get_placement_group', pg))
    monkeypatch.setattr(placement, 'remove_placement_group', lambda group: call('remove_placement_group', None))
    monkeypatch.setattr(
        placement, 'placement_group_table', lambda group: call('placement_group_table', {'state': 'REMOVED'}))
    pool = DevicePool('test', allocation_timeout=1, shutdown_timeout=.02)
    operation = asyncio.create_task(pool.remove('owned'))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        with pytest.raises(TimeoutError):
            await operation
        assert not returned.is_set(), 'Placement group RPC blocked the close deadline'
    finally:
        resume.set()
        await asyncio.gather(operation, return_exceptions=True)
        await asyncio.to_thread(returned.wait, 3)
