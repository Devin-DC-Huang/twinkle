# Copyright (c) ModelScope Contributors. All rights reserved.
"""Full-mode exclusivity must be one atomic state transition, not check-then-register."""
from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from starlette.requests import Request

import twinkle.protocol.types as types
from twinkle.server.exceptions import FullModeBusyError
from twinkle.server.model import twinkle_handlers
from twinkle.server.model.app import ModelManagement
from twinkle.server.model.twinkle_handlers import _register_model_twinkle_routes
from twinkle.server.session_resource.base import SessionResourceMixin


class _ResourceStub:
    """The resource-registry slice of ModelManagement, without Ray Serve."""

    is_full_mode = True
    _resource_type = 'Model'
    find_active_resource = SessionResourceMixin.find_active_resource
    register_resource = SessionResourceMixin.register_resource
    unregister_resource = SessionResourceMixin.unregister_resource
    set_resource_state = SessionResourceMixin.set_resource_state
    _validate_registration = SessionResourceMixin._validate_registration
    _create_resource_record = SessionResourceMixin._create_resource_record
    assert_full_mode_available = ModelManagement.assert_full_mode_available
    claim_full_mode_resource = ModelManagement.claim_full_mode_resource

    def __init__(self):
        self._resource_records = {}


@pytest.mark.asyncio
async def test_concurrent_full_mode_claims_have_exactly_one_winner():
    stub = _ResourceStub()

    async def claim(name):
        # The old check-then-register split had this await between the two steps.
        await asyncio.sleep(0)
        stub.claim_full_mode_resource(name, 'token', 'session')

    results = await asyncio.gather(claim('tenant-a'), claim('tenant-b'), return_exceptions=True)
    assert sum(isinstance(result, FullModeBusyError) for result in results) == 1
    assert len(stub._resource_records) == 1


def test_same_tenant_may_reclaim_its_own_resource():
    stub = _ResourceStub()
    stub.claim_full_mode_resource('tenant-a', 'token', 'session')
    stub.claim_full_mode_resource('tenant-a', 'token', 'session')

    with pytest.raises(FullModeBusyError):
        stub.claim_full_mode_resource('tenant-b', 'token', 'session')


class _AsyncState:

    def __init__(self):
        self.registered = []
        self.unloaded = []

    async def register_model(self, config, *, token, model_id, replica_id, session_id):
        # The exact window the old check-then-register split raced through.
        await asyncio.sleep(0)
        self.registered.append(model_id)

    async def unload_model(self, model_id):
        self.unloaded.append(model_id)


class _RunManager:

    def save(self, adapter_name, run_config):
        return None


class _FullModeManagement(_ResourceStub):
    """The handler-level slice of ModelManagement needed by add_adapter_to_model."""

    base_model = 'base-model'
    replica_id = 'replica'

    def __init__(self):
        super().__init__()
        self.state = _AsyncState()

    async def _on_request_start(self, request):
        return 'token'

    async def submit_and_peek(self, factory, **kwargs):
        from twinkle.protocol.types.lifecycle import TaskEnvelope
        return TaskEnvelope(request_id='req', status='completed', result=await factory())


def _request():
    request = Request({'type': 'http', 'headers': []})
    request.state.session_id = 'session'
    request.state.request_id = 'req'
    return request


@pytest.mark.asyncio
async def test_concurrent_handler_creates_have_exactly_one_winner(monkeypatch):
    """Two real handler calls, interleaved inside register_model, must not both win."""
    management = _FullModeManagement()
    monkeypatch.setattr(twinkle_handlers, 'create_training_run_manager', lambda *args, **kwargs: _RunManager())
    app = FastAPI()
    _register_model_twinkle_routes(app, lambda: management)
    endpoint = next(route.endpoint for route in app.routes
                    if getattr(route, 'path', None) == '/twinkle/add_adapter_to_model')

    results = await asyncio.gather(
        *(endpoint(_request(), types.AddAdapterRequest(adapter_name=name), management)
          for name in ('adapter-a', 'adapter-b')),
        return_exceptions=True)

    assert sum(isinstance(result, FullModeBusyError) for result in results) == 1
    assert len(management._resource_records) == 1
    # The loser released the model record it wrote before the claim failed.
    assert len(management.state.unloaded) == 1
