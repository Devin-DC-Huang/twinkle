# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ray/Serve resource effects shared by training and sampling owners."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from copy import deepcopy

from twinkle.server.env_propagation import build_propagated_env_vars
from .allocation import DevicePool

logger = logging.getLogger(__name__)


class RayRuntime:
    """Own deployment threads and certify worker/PG death before releasing capacity."""

    def __init__(self, namespace, *, allocation_timeout, shutdown_timeout):
        self.namespace = namespace
        self.pool = DevicePool(namespace, allocation_timeout=allocation_timeout, shutdown_timeout=shutdown_timeout)
        self.deployments = {}
        self.cleanup_tasks = set()

    def deployment_options(self, options, name, env_vars=None):
        options = deepcopy(options)
        options.pop('autoscaling_config', None)
        options['num_replicas'] = 1
        actor = options.setdefault('ray_actor_options', {})
        actor.setdefault('num_cpus', 0.1)
        runtime_env = actor.setdefault('runtime_env', {})
        env = build_propagated_env_vars()
        for key in ('PYTHONPATH', 'TWINKLE_DEFAULT_SAVE_DIR', 'TMPDIR', 'OMP_NUM_THREADS'):
            if key in os.environ:
                env.setdefault(key, os.environ[key])
        env.update(runtime_env.get('env_vars', {}))
        env.update(env_vars or {})
        env.update(TWINKLE_RAY_NAMESPACE=self.namespace, TWINKLE_RAY_REGISTRY_NAME=name + '-registry')
        runtime_env['env_vars'] = env
        return options

    async def deploy(self, identity, name, app, route_prefix, cleanup):
        from ray import serve
        deployment = asyncio.create_task(asyncio.to_thread(serve.run, app, name=name, route_prefix=route_prefix))
        self.deployments[identity] = deployment
        try:
            await asyncio.shield(deployment)
        except asyncio.CancelledError:

            async def clean_late():
                try:
                    await deployment
                except Exception:
                    pass
                await cleanup()

            task = asyncio.create_task(clean_late())
            self.cleanup_tasks.add(task)

            def done(completed):
                self.cleanup_tasks.discard(completed)
                if not completed.cancelled():
                    error = completed.exception()
                    if error is not None:
                        logger.error(
                            'Late deployment cleanup failed', exc_info=(type(error), error, error.__traceback__))

            task.add_done_callback(done)
            raise
        finally:
            if deployment.done():
                self.deployments.pop(identity, None)

    async def wait_for_cleanup(self):
        while self.cleanup_tasks:
            await asyncio.gather(*tuple(self.cleanup_tasks), return_exceptions=True)

    @staticmethod
    def handle(name):
        from ray import serve
        return serve.get_app_handle(name)

    async def shutdown(self, identity, name, *, observe_removal=None):
        import ray
        from ray import serve
        from ray.util import list_named_actors
        deployment = self.deployments.get(identity)
        if deployment is not None:
            try:
                await asyncio.shield(deployment)
            except Exception:
                pass
            self.deployments.pop(identity, None)
        # Both features give workers the controller identity, not a Serve replica ID.
        actors = []
        for row in await asyncio.to_thread(list_named_actors, all_namespaces=True):
            if row['namespace'] != self.namespace or identity not in row['name']:
                continue
            try:
                # Ray may import the worker class when resolving its handle.
                # Keep imports and GCS calls outside the controller event loop
                # so its shutdown deadline and other instances remain live.
                handle = await asyncio.to_thread(ray.get_actor, row['name'], namespace=self.namespace)
            except ValueError:
                continue
            actors.append(handle)
            await asyncio.to_thread(ray.kill, handle, no_restart=True)
        await asyncio.to_thread(serve.delete, name)
        started, status = time.monotonic(), 'error'
        try:
            await self.pool.remove(name)
            status = 'success'
        finally:
            if observe_removal is not None:
                observe_removal(time.monotonic() - started, status)
        for handle in actors:
            try:
                await asyncio.wait_for(handle.__ray_ready__.remote(), 5)
            except ray.exceptions.RayActorError:
                continue
            raise RuntimeError('Owned worker death could not be confirmed')

    async def recover_orphans(self, decode, shutdown):
        """Decode only this owner's names; never sweep another service's resources."""
        from ray import serve
        from ray.util.placement_group import placement_group_table
        names = set(serve.status().applications)
        names.update(row.get('name', '') for row in placement_group_table().values() if row.get('state') != 'REMOVED')
        for name in sorted(names):
            instance = decode(name)
            if instance is not None:
                await asyncio.wait_for(shutdown(instance), self.pool.shutdown_timeout)
