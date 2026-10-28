# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client for the platform-provided Service Mesh endpoint."""
from __future__ import annotations

import asyncio
import httpx
from contextlib import asynccontextmanager
from typing import AsyncIterator

from twinkle.server.common.tunnel import TunnelRequest, TunnelResponse

HEARTBEAT_PATHS = frozenset({'/api/v1/session_heartbeat', '/api/v1/twinkle/session_heartbeat'})
_QUERY_PATHS = frozenset({
    '/api/v1/models',
    '/api/v1/get_server_capabilities',
    '/api/v1/twinkle/get_server_capabilities',
    '/api/v1/twinkle/capacity_info',
    '/api/v1/twinkle/healthz',
})
_MAX_INFLIGHT_REQUESTS = 100


class ServiceMeshError(Exception):
    """The mesh returned a response that cannot be restored for the client."""


class ServiceMeshProxy:
    """Send Twinkle requests to one configured GPU service through Service Mesh."""

    def __init__(
        self,
        service_id: str,
        *,
        mesh_url: str = 'http://127.0.0.1:8880/api/v2/inference',
        timeout_seconds: float = 620.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._service_id = service_id
        self._mesh_url = mesh_url
        # The long timeout is for training responses, not connection admission.
        self._timeout = httpx.Timeout(timeout_seconds, connect=5.0, write=30.0, pool=3.0)
        self._client = client
        # Preserve the old shared pool's admission bound without sharing its
        # cancellation-sensitive connection state between independent requests.
        self._slots = asyncio.Semaphore(_MAX_INFLIGHT_REQUESTS)

    @asynccontextmanager
    async def _request_client(self) -> AsyncIterator[httpx.AsyncClient]:
        try:
            async with asyncio.timeout(self._timeout.pool):
                await self._slots.acquire()
        except TimeoutError as exc:
            raise httpx.PoolTimeout('CPU Gateway forwarding capacity is full.') from exc

        try:
            if self._client is not None:
                # Injected clients (e.g. in-process tests) belong to the caller.
                yield self._client
            else:
                # httpcore 1.0.9 may retain an unconnected pool slot after task
                # cancellation. Dispose of that pool with this request instead
                # of poisoning every later request to the long-lived gateway.
                async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
                    yield client
        finally:
            self._slots.release()

    async def forward(self, request_id: str, tunnel_request: TunnelRequest) -> TunnelResponse:
        timeout = self._timeout
        if tunnel_request.method == 'POST' and tunnel_request.path in HEARTBEAT_PATHS:
            timeout = httpx.Timeout(min(self._timeout.read, 10.0), connect=5.0, pool=3.0)
        elif tunnel_request.method == 'GET' and tunnel_request.path in _QUERY_PATHS:
            timeout = httpx.Timeout(min(self._timeout.read, 15.0), connect=5.0, pool=3.0)

        async with self._request_client() as client:
            response = await client.post(
                self._mesh_url,
                timeout=timeout,
                json={
                    'header': {
                        'request_id': request_id,
                        'service_id': self._service_id,
                    },
                    'payload': {
                        'input': tunnel_request.model_dump(mode='json'),
                        'parameters': {},
                    },
                },
            )
        response.raise_for_status()

        try:
            payload = response.json()
            header = payload['header']
            if header.get('status_code') != 200:
                raise ServiceMeshError(header.get('status_message', 'Service Mesh request failed.'))
            return TunnelResponse.model_validate(payload['payload']['output'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceMeshError('Invalid Service Mesh response.') from exc
