# Copyright (c) ModelScope Contributors. All rights reserved.
"""Aggregate plain instance snapshots without depending on training state or scheduling."""


class TrainingInstanceMetrics:

    def __init__(self, metrics):
        self.metrics = metrics
        self._instance_keys = set()
        self._resource_keys = set()

    def report(self, snapshots):
        counts, resources = {}, {}
        for tags, device_type, device_count, group_count in snapshots:
            key = tuple(tags[name] for name in ('backend', 'mode', 'state', 'resident'))
            counts[key] = counts.get(key, 0) + 1
            resource_key = tags['backend'], tags['mode'], device_type
            devices, groups = resources.get(resource_key, (0, 0))
            resources[resource_key] = devices + device_count, groups + group_count
        for key in self._instance_keys | counts.keys():
            tags = dict(zip(('backend', 'mode', 'state', 'resident'), key))
            self.metrics.instances_active.set(counts.get(key, 0), tags=tags)
        for key in self._resource_keys | resources.keys():
            tags = dict(zip(('backend', 'mode', 'device_type'), key))
            devices, groups = resources.get(key, (0, 0))
            self.metrics.devices_allocated.set(devices, tags=tags)
            self.metrics.placement_groups_active.set(groups, tags=tags)
        self._instance_keys = set(counts)
        self._resource_keys = set(resources)


def quarantine_reason(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, TimeoutError):
        return 'timeout'
    if 'worker death' in message:
        return 'worker_death'
    if 'placement group' in message:
        return 'pg_removal'
    return 'other'
