# Copyright (c) ModelScope Contributors. All rights reserved.
"""Initialization failures and backoff during automatic controller restarts."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from twinkle.server.training.actor import TrainingControllerActor


@pytest.mark.asyncio
async def test_failed_initialization_restarts_only_after_backoff(monkeypatch):
    cls = TrainingControllerActor.__ray_metadata__.modified_class
    gate, sleeping = asyncio.Event(), asyncio.Event()
    delays = []
    handle = object()
    kill = Mock()
    get_actor = Mock(return_value=handle)
    original_sleep = asyncio.sleep

    async def sleep(delay):
        delays.append(delay)
        sleeping.set()
        await gate.wait()

    monkeypatch.setattr('twinkle.server.training.actor.asyncio.sleep', sleep)
    monkeypatch.setattr('twinkle.server.training.actor.ray.get_actor', get_actor)
    monkeypatch.setattr('twinkle.server.training.actor.ray.kill', kill)
    actor = SimpleNamespace(core=SimpleNamespace(initialize=AsyncMock(side_effect=RuntimeError('allocation failed'))),
                            config=SimpleNamespace(sweep_interval_s=7), namespace='initialization-test', _monitors=[])
    actor._restart_after_failure = lambda: cls._restart_after_failure(actor)
    with pytest.raises(RuntimeError, match='allocation failed'):
        await cls._initialize(actor)
    await sleeping.wait()
    assert delays == [7] and not actor._monitors
    kill.assert_not_called()
    gate.set()
    await actor._restart_task
    await original_sleep(0)
    get_actor.assert_called_once_with('training-controller-initialization-test', namespace='initialization-test')
    kill.assert_called_once_with(handle, no_restart=False)


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_restart():
    cls = TrainingControllerActor.__ray_metadata__.modified_class
    wait = asyncio.Event()
    restart = asyncio.create_task(wait.wait())
    actor = SimpleNamespace(core=SimpleNamespace(shutdown=AsyncMock()), _monitors=[], _restart_task=restart)
    await cls.shutdown(actor)
    assert restart.cancelled()
    actor.core.shutdown.assert_awaited_once()
