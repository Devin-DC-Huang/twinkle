# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client for the platform-provided Service Mesh endpoint."""
from __future__ import annotations

import httpx

from twinkle.server.common.tunnel import TunnelRequest, TunnelResponse

HEARTBEAT_PATHS = frozenset({'/api/v1/session_heartbeat', '/api/v1/twinkle/session_heartbeat'})
_QUERY_PATHS = frozenset({
    '/api/v1/models',
    '/api/v1/get_server_capabilities',
    '/api/v1/twinkle/get_server_capabilities',
    '/api/v1/twinkle/capacity_info',
    '/api/v1/twinkle/healthz',
})


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
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=self._timeout,
            trust_env=False,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def forward(self, request_id: str, tunnel_request: TunnelRequest) -> TunnelResponse:
        timeout = self._timeout
        if tunnel_request.method == 'POST' and tunnel_request.path in HEARTBEAT_PATHS:
            timeout = httpx.Timeout(min(self._timeout.read, 10.0), connect=5.0, pool=3.0)
        elif tunnel_request.method == 'GET' and tunnel_request.path in _QUERY_PATHS:
            timeout = httpx.Timeout(min(self._timeout.read, 15.0), connect=5.0, pool=3.0)

        response = await self._client.post(
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
