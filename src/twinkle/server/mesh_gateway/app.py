# Copyright (c) ModelScope Contributors. All rights reserved.
"""Public Twinkle HTTP gateway backed by a configured Service Mesh target."""
from __future__ import annotations

import asyncio
import httpx
import json
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from twinkle.server.common.tunnel import TunnelRequest, TunnelResponse
from twinkle.utils.logger import get_logger
from .proxy import ServiceMeshError, ServiceMeshProxy

logger = get_logger()


def create_mesh_gateway_app(
    *,
    gpu_service_id: str,
    mesh_url: str = 'http://127.0.0.1:8880/api/v2/inference',
    timeout_seconds: float = 620.0,
    proxy: ServiceMeshProxy | None = None,
) -> FastAPI:
    """Create the CPU-side public gateway for one GPU training service."""
    mesh_proxy = proxy or ServiceMeshProxy(
        gpu_service_id,
        mesh_url=mesh_url,
        timeout_seconds=timeout_seconds,
    )

    app = FastAPI(
        title='Twinkle Service Mesh Gateway',
        description='Public Twinkle and Tinker gateway for one GPU service',
    )

    @app.get('/health')
    async def health() -> JSONResponse:
        return JSONResponse({'status': 'healthy'})

    @app.api_route('/api/v1/{path:path}', methods=['GET', 'POST', 'DELETE'])
    async def forward(path: str, request: Request) -> Response:
        request_id = request.headers.get('x-request-id') or str(uuid.uuid4())
        try:
            body = await _request_body(request)
            tunnel_request = TunnelRequest(
                method=request.method,
                path=f'/api/v1/{path}',
                query=dict(request.query_params),
                headers=dict(request.headers),
                body=body,
            )
        except (json.JSONDecodeError, ValueError):
            return JSONResponse({'detail': 'Request body must be JSON.'}, status_code=400)

        started_at = time.monotonic()
        logger.debug('Mesh forwarding started request_id=%s method=%s path=%s', request_id, request.method,
                     tunnel_request.path)
        try:
            tunnel_response = await _forward_until_disconnect(mesh_proxy, request, request_id, tunnel_request)
        except (httpx.HTTPError, ServiceMeshError) as exc:
            if isinstance(exc, httpx.PoolTimeout):
                status_code, detail = 503, 'CPU Gateway connection pool is busy. Try again later.'
            elif isinstance(exc, httpx.TimeoutException):
                status_code, detail = 504, 'GPU service timed out.'
            else:
                status_code, detail = 502, 'GPU service is unavailable.'
            logger.warning(
                'Mesh forwarding failed request_id=%s method=%s path=%s service_id=%s '
                'error_type=%s status=%s elapsed_seconds=%.3f',
                request_id,
                request.method,
                tunnel_request.path,
                gpu_service_id,
                type(exc).__name__,
                status_code,
                time.monotonic() - started_at,
            )
            return JSONResponse({'detail': detail}, status_code=status_code)

        if tunnel_response is None:
            logger.info('Mesh forwarding cancelled after client disconnect request_id=%s path=%s elapsed_seconds=%.3f',
                        request_id, tunnel_request.path,
                        time.monotonic() - started_at)
            return Response(status_code=499)

        logger.debug('Mesh forwarding completed request_id=%s status=%s elapsed_seconds=%.3f', request_id,
                     tunnel_response.status_code,
                     time.monotonic() - started_at)

        return JSONResponse(
            tunnel_response.body,
            status_code=tunnel_response.status_code,
            headers=tunnel_response.headers,
        )

    return app


async def _request_body(request: Request):
    if not await request.body():
        return None
    return await request.json()


async def _forward_until_disconnect(
    proxy: ServiceMeshProxy,
    request: Request,
    request_id: str,
    tunnel_request: TunnelRequest,
) -> TunnelResponse | None:
    # The request body has already been consumed, so this is its only remaining reader.
    async def wait_for_disconnect():
        while (await request.receive())['type'] != 'http.disconnect':
            pass

    forwarding = asyncio.create_task(proxy.forward(request_id, tunnel_request))
    disconnected = asyncio.create_task(wait_for_disconnect())
    try:
        done, _ = await asyncio.wait((forwarding, disconnected), return_when=asyncio.FIRST_COMPLETED)
        if forwarding in done:
            return await forwarding
        # Release the CPU-side HTTP request; this does not roll back GPU training.
        return None
    finally:
        for task in (forwarding, disconnected):
            task.cancel()
        await asyncio.gather(forwarding, disconnected, return_exceptions=True)
