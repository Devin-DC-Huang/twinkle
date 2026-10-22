# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stable CPU ingress for an owned sampling pool; existing sampler wire is forwarded intact."""
from __future__ import annotations

import asyncio
import json
import uuid
from copy import deepcopy
from fastapi import Depends, Request, Response
from fastapi.responses import StreamingResponse

from twinkle.server.checkpoint import create_checkpoint_manager
from twinkle.server.deployment import bind_deployment, build_deployment_app
from twinkle.server.middleware.auth import get_token_from_request
from twinkle.server.state import get_server_state
from .controller import create_controller_actor, sampler_controller_name, sampling_pool_id
from .weights import is_full_checkpoint


class SamplerRouter:

    async def __init__(self, namespace, pool_id):
        import ray
        self.state = get_server_state()
        self.controller = ray.get_actor(sampler_controller_name(namespace, pool_id), namespace=namespace)

    async def forward(self, request, endpoint):
        body = await request.body()
        payload = {
            'endpoint': endpoint,
            'body': body,
            'method': request.method,
            'headers': list(request.headers.raw),
            'query_string': request.scope.get('query_string', b'')
        }
        full_path = None
        if endpoint in ('twinkle/sample', 'twinkle/sample_stream', 'twinkle/sample_to_data_plane', 'tinker/asample'):
            try:
                data = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            # The existing handler remains responsible for its validation/wire errors.
            path = data.get('adapter_uri') or data.get('model_path')
            if not path and endpoint == 'tinker/asample' and data.get('sampling_session_id'):
                session = await self.state.get_sampling_session(data['sampling_session_id'])
                path = session.get('model_path') if session else None
            if path:
                client_type = 'tinker' if endpoint.startswith('tinker/') else 'twinkle'
                _, resolved = create_checkpoint_manager(
                    get_token_from_request(request), client_type=client_type).parse_adapter_uri(path)
                if await asyncio.to_thread(is_full_checkpoint, resolved):
                    full_path = resolved
        stream = endpoint == 'twinkle/sample_stream'
        queue = None
        if stream:
            from ray.util.queue import Queue
            queue = Queue(maxsize=128)
        try:
            reply = await self.controller.request.remote(payload, queue, full_path)
        except BaseException:
            if queue is not None:
                queue.shutdown(force=True)
            raise
        headers = {
            key.decode(): value.decode()
            for key, value in reply['headers'] if key.lower() not in (b'content-length', b'transfer-encoding')
        }
        if not stream:
            return Response(reply['body'], status_code=reply['status'], headers=headers)

        async def generate():
            try:
                while True:
                    item = await queue.get_async()
                    if item is None:
                        return
                    yield item
            finally:
                queue.shutdown(force=True)

        return StreamingResponse(generate(), status_code=reply['status'], headers=headers)


def build_sampler_router(namespace, pool_id, deploy_options):

    def routes(app, get_self):

        @app.get('/healthz')
        async def health(self=Depends(get_self)):
            diagnostic = await self.controller.diagnostics.remote()
            return {'healthy': diagnostic['ready'], **diagnostic}

        @app.post('/twinkle/{endpoint}')
        async def native(request: Request, endpoint: str, self=Depends(get_self)):
            return await self.forward(request, f'twinkle/{endpoint}')

        @app.post('/tinker/{endpoint}')
        async def tinker(request: Request, endpoint: str, self=Depends(get_self)):
            return await self.forward(request, f'tinker/{endpoint}')

    app = build_deployment_app('SamplerRouter', routes)
    return bind_deployment(
        app, SamplerRouter, deploy_options, deployment_name='SamplerRouter', bind_args=(namespace, pool_id))


def deploy_sampler_pool(spec, namespace, budgets, env):
    """Driver-only composition. Each compute instance owns its external allocation."""
    import ray
    from ray import serve
    options = deepcopy(spec.deployments[0]) if spec.deployments else {}
    options.pop('name', None)
    options.pop('autoscaling_config', None)
    options['num_replicas'] = 1
    actor_options = options.setdefault('ray_actor_options', {})
    actor_options.setdefault('num_cpus', 0.1)
    runtime_env = actor_options.setdefault('runtime_env', {})
    runtime_env['env_vars'] = {**env, **runtime_env.get('env_vars', {}), 'TWINKLE_STATE_RAY_NAMESPACE': namespace}
    pool_id = sampling_pool_id(spec.route_prefix)
    name = sampler_controller_name(namespace, pool_id)
    controller = create_controller_actor().options(
        name=name, namespace=namespace, runtime_env=runtime_env).remote(spec.args.model_dump(), namespace, pool_id,
                                                                        options, budgets)
    try:
        ray.get(controller.initialize.remote())
        app = build_sampler_router(namespace, pool_id, options)
        serve.run(app, name=spec.name, route_prefix=spec.route_prefix)
    except BaseException:
        try:
            ray.get(controller.shutdown.remote())
        finally:
            ray.kill(controller, no_restart=True)
        raise
    return controller
