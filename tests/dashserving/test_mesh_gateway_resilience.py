# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU Gateway failure isolation, with a real TCP connection-pool regression."""
from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
import pytest

from twinkle.server.common.tunnel import TunnelRequest
from twinkle.server.mesh_gateway import create_mesh_gateway_app
from twinkle.server.mesh_gateway.logging import HeartbeatAccessLogFilter
from twinkle.server.mesh_gateway.proxy import ServiceMeshProxy


def _mesh_response():
    return {
        'header': {'status_code': 200},
        'payload': {'output': {'status_code': 200, 'headers': {}, 'body': {'ok': True}}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(('method', 'path', 'read_timeout'), [
    ('POST', '/api/v1/session_heartbeat', 10.0),
    ('POST', '/api/v1/twinkle/session_heartbeat', 10.0),
    ('GET', '/api/v1/models', 15.0),
    ('POST', '/api/v1/forward_backward', 620.0),
    ('POST', '/api/v1/model/Qwen/model/twinkle/save', 620.0),
    ('GET', '/api/v1/twinkle/healthz/deep', 620.0),
])
async def test_long_training_timeout_does_not_apply_to_pool_or_heartbeat(method, path, read_timeout):
    def respond(request):
        observed = request.extensions['timeout']
        assert observed['read'] == read_timeout
        assert observed['connect'] == 5.0
        assert observed['pool'] == 3.0
        payload = json.loads(request.content)
        assert payload['header'] == {'request_id': 'request-id', 'service_id': 'gpu-service'}
        assert payload['payload']['input']['path'] == path
        return httpx.Response(200, json=_mesh_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as upstream:
        proxy = ServiceMeshProxy('gpu-service', client=upstream)
        result = await proxy.forward('request-id', TunnelRequest(method=method, path=path))
        assert result.body == {'ok': True}


@pytest.mark.asyncio
@pytest.mark.parametrize(('error', 'status'), [
    (httpx.PoolTimeout, 503),
    (httpx.ConnectTimeout, 504),
    (httpx.ReadTimeout, 504),
    (httpx.ConnectError, 502),
])
async def test_gateway_reports_failure_without_retrying_training(error, status):
    calls = []

    def fail(request):
        calls.append(request)
        raise error('upstream unavailable', request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as upstream:
        app = create_mesh_gateway_app(
            gpu_service_id='gpu-service', proxy=ServiceMeshProxy('gpu-service', client=upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
            response = await client.post('/api/v1/optim_step', json={})
            assert response.status_code == status
            assert len(calls) == 1
            assert (await client.get('/health')).status_code == 200


@pytest.mark.asyncio
async def test_real_saturated_pool_fails_fast_and_recovers():
    """Exercise HTTPcore admission; MockTransport cannot exhaust a TCP pool."""
    busy, release = asyncio.Event(), asyncio.Event()
    body = json.dumps(_mesh_response()).encode()

    async def handle(reader, writer):
        try:
            headers = await reader.readuntil(b'\r\n\r\n')
            length = next(int(line.split(b':', 1)[1]) for line in headers.split(b'\r\n')
                          if line.lower().startswith(b'content-length:'))
            await reader.readexactly(length)
            busy.set()
            await release.wait()
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
                         b'Connection: close\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    async with server, httpx.AsyncClient(limits=httpx.Limits(max_connections=1)) as upstream:
        proxy = ServiceMeshProxy('gpu-service', mesh_url=f'http://127.0.0.1:{port}/inference', client=upstream)
        app = create_mesh_gateway_app(gpu_service_id='gpu-service', proxy=proxy)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
            training = asyncio.create_task(client.post('/api/v1/optim_step', json={}))
            try:
                await asyncio.wait_for(busy.wait(), timeout=5)
                assert (await client.get('/health')).status_code == 200
                started_at = time.monotonic()
                response = await asyncio.wait_for(client.get('/api/v1/models'), timeout=5)
                assert response.status_code == 503
                assert time.monotonic() - started_at < 5
            finally:
                release.set()
                await training
            assert training.result().status_code == 200
            assert (await client.get('/api/v1/models')).status_code == 200


def _access_record(path, status=404):
    return logging.LogRecord(
        'uvicorn.access', logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
        ('client', 'POST', path, '1.1', status), None)


def test_only_repeated_heartbeat_404_access_logs_are_sampled(monkeypatch):
    from twinkle.server.mesh_gateway import logging as gateway_logging

    monkeypatch.setattr(gateway_logging.time, 'monotonic', lambda: 10.0)
    access_filter = HeartbeatAccessLogFilter()
    assert access_filter.filter(_access_record('/api/v1/session_heartbeat'))
    assert not access_filter.filter(_access_record('/api/v1/session_heartbeat'))
    assert not access_filter.filter(_access_record('/api/v1/twinkle/session_heartbeat'))
    assert access_filter.filter(_access_record('/api/v1/session_heartbeat', 500))
    assert access_filter.filter(_access_record('/api/v1/session_heartbeat', 200))
    assert access_filter.filter(_access_record('/api/v1/get_info'))
    monkeypatch.setattr(gateway_logging.time, 'monotonic', lambda: 71.0)
    summary = _access_record('/api/v1/session_heartbeat')
    assert access_filter.filter(summary)
    assert 'suppressed 2 similar heartbeat 404' in summary.getMessage()


@pytest.mark.asyncio
async def test_unknown_session_response_is_not_hidden():
    def respond(request):
        result = _mesh_response()
        result['payload']['output'] = {'status_code': 404, 'headers': {}, 'body': {'detail': 'Unknown session'}}
        return httpx.Response(200, json=result)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as upstream:
        app = create_mesh_gateway_app(
            gpu_service_id='gpu-service', proxy=ServiceMeshProxy('gpu-service', client=upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
            response = await client.post('/api/v1/session_heartbeat', json={'session_id': 'expired'})
            assert response.status_code == 404
            assert response.json() == {'detail': 'Unknown session'}


@pytest.mark.asyncio
async def test_client_disconnect_cancels_cpu_forwarding():
    started, cancelled = asyncio.Event(), asyncio.Event()

    class WaitingProxy:
        async def forward(self, request_id, tunnel_request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    app = create_mesh_gateway_app(gpu_service_id='gpu-service', proxy=WaitingProxy())
    scope = {
        'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'GET',
        'scheme': 'http', 'path': '/api/v1/models', 'raw_path': b'/api/v1/models',
        'query_string': b'', 'headers': [], 'server': ('gateway', 80), 'client': ('client', 1234),
    }
    body_sent = False
    messages = []

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        await started.wait()
        return {'type': 'http.disconnect'}

    async def send(message):
        messages.append(message)

    await asyncio.wait_for(app(scope, receive, send), timeout=1)
    assert cancelled.is_set()
    assert messages[0]['status'] == 499
