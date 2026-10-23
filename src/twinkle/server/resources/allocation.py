# Copyright (c) ModelScope Contributors. All rights reserved.
"""Single-node device allocation; only the resource owner removes placement groups."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from twinkle import Platform
from twinkle.infra._ray.resource_manager import _gpu_placement_group_cpus
from twinkle.server.exceptions import RequestRejectedError


class CapacityUnavailableError(RequestRejectedError):
    error_code = 409


@dataclass(frozen=True)
class DeviceAllocation:
    name: str
    placement_group_id: str
    node_id: str
    devices: tuple[str, ...]


def _probe_allocation(resource: str, device_type: str):
    import os
    import ray
    context = ray.get_runtime_context()
    if resource == 'CPU':
        devices = []
    else:
        env = os.environ.get(Platform.get_platform(device_type).visible_device_env())
        devices = env.split(',') if env else context.get_accelerator_ids().get(resource, [])
    return context.get_node_id(), [str(device) for device in devices]


class DevicePool:

    def __init__(self, namespace: str, *, allocation_timeout: float, shutdown_timeout: float):
        self.namespace = namespace
        self.allocation_timeout = allocation_timeout
        self.shutdown_timeout = shutdown_timeout
        self._creating = {}

    def name(self, epoch: str, instance_id: str) -> str:
        return f'resources-{self.namespace}-{epoch}-{instance_id}'

    @staticmethod
    def _create_group(resource, count, name):
        import ray
        from ray.util.placement_group import placement_group
        candidates = [node for node in ray.nodes() if node.get('Alive') and node['Resources'].get(resource, 0) >= count]
        available = ray.available_resources()
        if not candidates or available.get(resource, 0) < count:
            raise CapacityUnavailableError(f'No single-node allocation available for {count} {resource} devices')
        cpus = min(_gpu_placement_group_cpus(int(node['Resources'].get('CPU', 0)), count) for node in candidates)
        bundle = {resource: count, 'CPU': count if resource == 'CPU' else cpus}
        return placement_group([bundle], strategy='STRICT_PACK', name=name)

    async def allocate(self, num_devices: int, device_type: str, epoch: str, instance_id: str) -> DeviceAllocation:
        import ray
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        if type(num_devices) is not int or num_devices <= 0:
            raise ValueError('num_devices must be a positive integer')
        resource = 'CPU' if device_type == 'cpu' else Platform.get_platform(device_type).__name__
        count = num_devices
        name = self.name(epoch, instance_id)
        creating = asyncio.create_task(asyncio.to_thread(self._create_group, resource, count, name))
        self._creating[name] = creating
        try:
            async with asyncio.timeout(self.allocation_timeout):
                pg = await asyncio.shield(creating)
                await pg.ready()
            options = {'num_cpus': 0.01, 'scheduling_strategy': PlacementGroupSchedulingStrategy(placement_group=pg)}
            if resource == 'GPU':
                options['num_gpus'] = count
            elif resource != 'CPU':
                options['resources'] = {resource: count}
            node, devices = await asyncio.wait_for(
                ray.remote(_probe_allocation).options(**options).remote(resource, device_type), self.shutdown_timeout)
            if resource == 'CPU':
                devices = [str(index) for index in range(count)]
            if len(devices) != count or len(set(devices)) != count:
                raise RuntimeError('Ray allocation did not return the requested unique device IDs')
            return DeviceAllocation(name, pg.id.hex(), node, tuple(devices))
        except BaseException as exc:
            # Cancellation cannot stop a synchronous GCS create. Keep its task
            # owned until removal has observed the late result; a close timeout
            # leaves the enclosing instance quarantined for a later retry.
            await self.remove(name)
            if isinstance(exc, TimeoutError):
                raise CapacityUnavailableError(
                    'Single-node allocation could not be confirmed; retry a new create later.') from exc
            raise
        finally:
            if creating.done() and self._creating.get(name) is creating:
                self._creating.pop(name)

    async def remove(self, name: str) -> None:
        from ray.util.placement_group import get_placement_group, placement_group_table, remove_placement_group
        try:
            async with asyncio.timeout(self.shutdown_timeout):
                creating = self._creating.get(name)
                if creating is not None:
                    try:
                        await asyncio.shield(creating)
                    except Exception:
                        pass
                    if self._creating.get(name) is creating:
                        self._creating.pop(name)
                try:
                    pg = await asyncio.to_thread(get_placement_group, name)
                except ValueError:
                    return
                await asyncio.to_thread(remove_placement_group, pg)
                while True:
                    if (await asyncio.to_thread(placement_group_table, pg)).get('state') == 'REMOVED':
                        return
                    await asyncio.sleep(0.05)
        except TimeoutError as exc:
            raise TimeoutError(f'Placement group removal was not confirmed: {name}') from exc
