# Copyright (c) ModelScope Contributors. All rights reserved.
"""Controller-owned persistence; other key-spaces are accessed through ServerState."""
from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

from .backend.base import StateBackend
from .models import FutureFailureRecord
from .training_records import CreateTraining, InstanceRecord, TrainingRecord

if TYPE_CHECKING:
    from .server_state import ServerState


class TrainingStore:

    def __init__(self, state: ServerState, namespace: str, backend: StateBackend):
        self.state = state
        self.backend = backend
        self.prefix = f'training::{namespace}::'

    def key(self, kind: str, identity: str) -> str:
        return f'{self.prefix}{kind}::{identity}'

    @staticmethod
    def create_identity(request: CreateTraining) -> tuple[str, str]:
        identity = json.dumps([request.client_type, request.token, request.session_id, request.create_key])
        fingerprint = json.dumps(request.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(identity.encode()).hexdigest(), hashlib.sha256(fingerprint.encode()).hexdigest()

    async def reserve(self, record: TrainingRecord) -> None:
        await self.state.register_model_record(record.training_id, record)

    async def save_training(self, record: TrainingRecord) -> None:
        await self.state.save_model_record(record.training_id, record)

    async def retire(self, record: TrainingRecord) -> None:
        await self.backend.set(
            self.key('terminal', record.training_id),
            record.model_dump(),
            ttl=max(1, int(self.state.expiration_timeout)))
        await self.state.delete_training_sequences(record.training_id)
        # Release the quota row last so a failed earlier step can be idempotently retried.
        await self.state.unload_model(record.training_id)

    async def save_instance(self, record: InstanceRecord) -> None:
        ttl = max(1, int(self.state.expiration_timeout)) if record.state == 'STOPPED' else None
        await self.backend.set(self.key('instance', record.instance_id), record.model_dump(), ttl=ttl)

    async def instances(self) -> list[InstanceRecord]:
        keys = await self.backend.keys(self.key('instance', '*'))
        return [InstanceRecord.model_validate(value) for value in await self.backend.mget(keys) if value is not None]

    async def trainings(self) -> list[TrainingRecord]:
        return await self.state.get_training_records(self.prefix)

    async def get_create(self, identity: str) -> dict[str, str] | None:
        return await self.backend.get(self.key('create', identity))

    async def save_create(self, identity: str, fingerprint: str, request_id: str, session_id: str) -> None:
        await self.backend.set(
            self.key('create', identity), {
                'fingerprint': fingerprint,
                'request_id': request_id,
                'session_id': session_id
            })

    async def cleanup_receipts(self) -> None:
        for key in await self.backend.keys(self.key('create', '*')):
            receipt = await self.backend.get(key)
            if receipt and await self.state.get_session_last_heartbeat(receipt['session_id']) is None:
                await self.backend.delete(key)

    async def future(self,
                     request_id: str,
                     status: str,
                     model_id: str | None,
                     *,
                     epoch: str | None = None,
                     result: Any = None,
                     failure: FutureFailureRecord | None = None,
                     deadline: float | None = None) -> None:
        await self.state.store_future_status(
            request_id, status, model_id, result=result, failure=failure, replica_id=epoch, absolute_deadline=deadline)

    async def own_future(self, request_id: str, token: str) -> None:
        await self.state.set_future_owner(request_id, token)

    async def begin_epoch(self, epoch: str) -> None:
        await self.backend.set(self.key('controller', 'active'), {'epoch': epoch, 'started_at': time.time()})
