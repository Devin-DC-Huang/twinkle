# Copyright (c) ModelScope Contributors. All rights reserved.
"""Full-mode exclusivity must be one atomic state transition, not check-then-register."""
from __future__ import annotations

import asyncio

import pytest

from twinkle.server.exceptions import FullModeBusyError
from twinkle.server.model.app import ModelManagement
from twinkle.server.session_resource.base import SessionResourceMixin


class _ResourceStub:
    """The resource-registry slice of ModelManagement, without Ray Serve."""

    is_full_mode = True
    _resource_type = 'Model'
    find_active_resource = SessionResourceMixin.find_active_resource
    register_resource = SessionResourceMixin.register_resource
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
