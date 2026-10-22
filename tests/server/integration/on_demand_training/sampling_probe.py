# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real Ray/Serve 2->4->2 allocation and death confirmation, without model kernels."""
import asyncio
import json
import os
import uuid

import ray
from ray import serve
from ray.util.placement_group import get_placement_group, placement_group_table
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from twinkle.server.config.application_spec import SamplerArgs
from twinkle.server.deployment import bind_deployment, build_deployment_app
from twinkle.server.resources.allocation import CapacityUnavailableError, DevicePool
from twinkle.server.sampler.controller import SamplerController
from twinkle.server.sampler.driver import SamplerDriver


@ray.remote(num_cpus=0.01, num_gpus=1)
class Worker:

    def device(self):
        return ray.get_runtime_context().get_accelerator_ids()['GPU']


class Compute:

    async def __init__(self, namespace, identity, pg):
        self.operations = {}
        self.worker = Worker.options(
            name=identity + '-worker',
            namespace=namespace,
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote()
        self.devices = await self.worker.device.remote()

    async def invoke(self, ticket, payload, stream_queue=None):
        self.operations[ticket] = asyncio.Event()
        return {'status': 200, 'headers': [], 'body': b'{"status":"pending"}'}

    async def wait_operation(self, ticket):
        await self.operations[ticket].wait()
        self.operations.pop(ticket)

    async def finish(self):
        for operation in self.operations.values():
            operation.set()

    async def pending_futures(self):
        return []

    async def runtime_identity(self):
        return {'unhealthy': False, 'devices': self.devices}


class Driver(SamplerDriver):

    async def start(self, instance):
        allocation = await self.runtime.pool.allocate(1, 'cuda', f'sampling_{self.pool_id}_{instance.epoch}',
                                                      instance.identity)
        instance.application, instance.devices = allocation.name, list(allocation.devices)
        app = bind_deployment(
            build_deployment_app('AllocationProbe', lambda app, get_self: None),
            Compute,
            self.runtime.deployment_options({}, allocation.name),
            deployment_name='AllocationCompute',
            bind_args=(self.namespace, instance.identity, get_placement_group(allocation.name)))
        await self.runtime.deploy(instance.identity, allocation.name, app, '/probe/' + instance.identity,
                                  lambda: self.shutdown(instance))


async def run(namespace):
    args = SamplerArgs(
        model_id='allocation-probe',
        sampler_type='mock',
        device_group={
            'ranks': 1,
            'device_type': 'cuda'
        },
        device_mesh={'dp_size': 1},
        min_replicas=2,
        max_replicas=4,
        target_ongoing_requests=1,
        idle_timeout_s=1)
    budgets = {'allocation_timeout_s': 5, 'startup_timeout_s': 600, 'shutdown_timeout_s': 180, 'sweep_interval_s': 1}
    driver = Driver(args, namespace, 'allocation', {}, budgets)
    clock = [0]
    pool = SamplerController(args, driver, budgets, clock=lambda: clock[0])
    training_pool = DevicePool(namespace, allocation_timeout=5, shutdown_timeout=180)
    training = []
    evidence = {'sizes': [], 'device_ids': []}
    try:
        # Two training-sized allocations plus four sampling instances share exactly eight GPUs.
        for index in range(2):
            training.append(await training_pool.allocate(2, 'cuda', 'probe_training', str(index)))
        await pool.initialize()
        evidence['sizes'].append(pool._owned_count())
        payload = {'endpoint': 'twinkle/sample', 'method': 'POST', 'headers': [], 'body': b'{}'}
        for _ in range(8):
            await pool.request(payload)
            await asyncio.gather(*tuple(pool.lifecycle.startups.values()))
        evidence['sizes'].append(pool._owned_count())
        all_devices = [device for allocation in training for device in allocation.devices]
        for instance in pool._ready_instances():
            identity = await driver.health(instance)
            assert identity['devices'] == instance.devices
            all_devices.extend(instance.devices)
        assert len(all_devices) == len(set(all_devices)) == 8
        evidence['device_ids'] = sorted(all_devices)
        try:
            await training_pool.allocate(1, 'cuda', 'probe_training', 'overflow')
        except CapacityUnavailableError:
            evidence['no_overcommit'] = True
        else:
            raise AssertionError('A ninth device allocation unexpectedly succeeded')
        clock[0] = 10
        await pool.sweep()
        assert pool._owned_count() == 4  # Public replies did not retire actual operations.
        await asyncio.gather(*(driver.handle(instance).finish.remote() for instance in pool._ready_instances()))
        await pool.lifecycle.join()
        clock[0] = 12
        await pool.sweep()
        await pool.lifecycle.join()
        evidence['sizes'].append(len(pool._ready_instances()))
        assert evidence['sizes'] == [2, 4, 2]
        old_names = {instance.application for instance in pool._ready_instances()}
        recovered_driver = Driver(args, namespace, 'allocation', {}, budgets)
        recovered = SamplerController(args, recovered_driver, budgets, clock=lambda: clock[0])
        try:
            await recovered.initialize()
            assert recovered.epoch != pool.epoch and len(recovered._ready_instances()) == 2
            assert not old_names.intersection(serve.status().applications)
            assert all(
                placement_group_table(get_placement_group(allocation.name))['state'] != 'REMOVED'
                for allocation in training)
            evidence['owned_orphan_recovery'] = True
        finally:
            await recovered.shutdown()
            await recovered_driver.runtime.wait_for_cleanup()
        await pool.shutdown()
        await driver.runtime.wait_for_cleanup()
        names = set(serve.status().applications)
        assert not any(name.startswith(driver.prefix) for name in names)
        assert not any(row['state'] != 'REMOVED' and row.get('name', '').startswith(driver.prefix)
                       for row in placement_group_table().values())
        from ray.util import list_named_actors
        assert not any(row['namespace'] == namespace and 'sampler_' in row['name']
                       for row in list_named_actors(all_namespaces=True))
        evidence['confirmed_release'] = True
        return evidence
    finally:
        await pool.shutdown()
        await driver.runtime.wait_for_cleanup()
        for allocation in training:
            await training_pool.remove(allocation.name)


def main():
    namespace = 'sampling_probe_' + uuid.uuid4().hex[:8]
    os.environ.update(
        TWINKLE_STATE_RAY_NAMESPACE=namespace, TWINKLE_PERSISTENCE_MODE='memory', TWINKLE_GPU_PG_CPUS_PER_PROC='1')
    ray.init(
        address='local',
        namespace=namespace,
        num_cpus=24,
        num_gpus=8,
        _temp_dir='/tmp/tws-' + namespace[-8:],
        include_dashboard=False)
    try:
        serve.start(http_options={'host': '127.0.0.1', 'port': 0})
        print(json.dumps(asyncio.run(run(namespace))), flush=True)
    finally:
        serve.shutdown()
        ray.shutdown()


if __name__ == '__main__':
    main()
