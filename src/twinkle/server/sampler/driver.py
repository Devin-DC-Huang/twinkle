# Copyright (c) ModelScope Contributors. All rights reserved.
"""Owned Ray/Serve effects for sampling instances; no training lifecycle dependency."""
from __future__ import annotations

import asyncio

from twinkle.server.config.application_spec import SamplerArgs
from twinkle.server.resources.runtime import RayRuntime
from twinkle.server.state import get_server_state
from twinkle.server.state.models import FutureFailureRecord


class SamplerDriver:

    def __init__(self, args: SamplerArgs, namespace: str, pool_id: str, options: dict, budgets: dict):
        self.args, self.namespace, self.pool_id = args, namespace, pool_id
        self.options, self.budgets = options, budgets
        self.runtime = RayRuntime(
            namespace,
            allocation_timeout=budgets['allocation_timeout_s'],
            shutdown_timeout=budgets['shutdown_timeout_s'])

    @property
    def prefix(self):
        return f'resources-{self.namespace}-sampling_{self.pool_id}_'

    async def start(self, instance):
        from ray.util.placement_group import get_placement_group

        from .app import build_sampler_app
        group = self.args.device_group
        device_type = str(group.get('device_type', 'cpu' if self.args.sampler_type == 'mock' else 'cuda')).lower()
        allocation = await self.runtime.pool.allocate(
            group.get('ranks', self.args.nproc_per_node), device_type, f'sampling_{self.pool_id}_{instance.epoch}',
            instance.identity)
        instance.application = allocation.name
        instance.devices = list(allocation.devices)
        options = self.runtime.deployment_options(self.options, allocation.name)
        values = self.args.model_dump(
            exclude={'min_replicas', 'max_replicas', 'idle_timeout_s', 'target_ongoing_requests', 'queue_config'})
        values['device_group'].setdefault('name', 'sampler')
        values['device_group'].setdefault('ranks', self.args.nproc_per_node)
        values['device_group'].setdefault('device_type', device_type)
        app = build_sampler_app(
            **values,
            queue_config=self.args.queue_config,
            deploy_options=options,
            sampling_context={
                'namespace': self.namespace,
                'registry_name': allocation.name + '-registry',
                'instance_id': instance.identity
            },
            placement_group=get_placement_group(allocation.name),
            allocation_devices=list(allocation.devices))
        await self.runtime.deploy(instance.identity, allocation.name, app,
                                  f'/api/v1/sampling-runtime/{instance.identity}', lambda: self.shutdown(instance))

    def handle(self, instance):
        return self.runtime.handle(instance.application)

    async def invoke(self, instance, ticket, payload, stream_queue=None):
        return await self.handle(instance).invoke.remote(ticket, payload, stream_queue)

    async def wait_operation(self, instance, ticket):
        await self.handle(instance).wait_operation.remote(ticket)

    async def load_weights(self, instance, path):
        await self.handle(instance).install_base_weights.remote(path)

    async def health(self, instance):
        return await asyncio.wait_for(self.handle(instance).runtime_identity.remote(), 5)

    async def shutdown(self, instance):
        name = instance.application or self.runtime.pool.name(f'sampling_{self.pool_id}_{instance.epoch}',
                                                              instance.identity)
        futures = []
        try:
            futures = await asyncio.wait_for(self.handle(instance).pending_futures.remote(), 5)
        except Exception:
            pass
        await self.runtime.shutdown(instance.identity, name)
        failure = FutureFailureRecord(
            reason_code='orphaned_replica',
            attribution='server',
            message='Sampler instance stopped; retry with the checkpoint path.')
        state = get_server_state()
        for request_id in futures:
            await state.store_future_status(request_id, 'failed', None, failure=failure)

    async def recover_orphans(self):
        import re
        from types import SimpleNamespace
        pattern = re.compile(re.escape(self.prefix) + r'([0-9a-f]{32})-(sampler_[0-9a-f]{32})$')

        def decode(name):
            match = pattern.fullmatch(name)
            if match:
                return SimpleNamespace(identity=match[2], epoch=match[1], application=name)

        await self.runtime.recover_orphans(decode, self.shutdown)
