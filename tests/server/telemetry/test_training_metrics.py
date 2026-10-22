# Copyright (c) ModelScope Contributors. All rights reserved.
"""Managed training metrics stay low-cardinality and survive registry resets."""
from collections import defaultdict
from types import SimpleNamespace

import pytest

from twinkle.server.telemetry import metrics
from twinkle.server.training.controller import TrainingController
from twinkle.server.telemetry.training_metrics import TrainingInstanceMetrics

_ALLOWED_ATTRIBUTES = {'backend', 'mode', 'state', 'status', 'reason', 'device_type', 'resident'}


@pytest.fixture(autouse=True)
def reset_registry():
    yield
    metrics.MetricsRegistry.reset()


class FakeInstrument:

    def __init__(self, name):
        self.name = name
        self.calls = []

    def add(self, value, attributes=None):
        self.calls.append((value, attributes or {}))

    def record(self, value, attributes=None):
        self.calls.append((value, attributes or {}))


class FakeMeter:

    def __init__(self):
        self.instruments = {}

    def _create(self, name, **kwargs):
        instrument = FakeInstrument(name)
        self.instruments[name] = instrument
        return instrument

    create_counter = _create
    create_histogram = _create
    create_up_down_counter = _create

    def create_observable_gauge(self, name, **kwargs):
        return self._create(name, **kwargs)


def build_registry(monkeypatch):
    meter = FakeMeter()
    monkeypatch.setattr(metrics, 'get_meter', lambda name: meter)
    metrics.MetricsRegistry.reset()
    return metrics.MetricsRegistry.get(), meter


def test_training_and_redis_instruments_are_registered(monkeypatch):
    registry, meter = build_registry(monkeypatch)
    training = registry.training_metrics()
    state = registry.state_backend_metrics()

    training.instances_active.set(2, tags={'backend': 'transformers', 'mode': 'lora', 'state': 'READY',
                                           'resident': 'false'})
    training.startup_seconds.observe(1.5, tags={'backend': 'transformers', 'mode': 'lora', 'status': 'success'})
    training.quarantined_total.inc(tags={'backend': 'megatron', 'mode': 'full', 'reason': 'timeout'})
    state.redis_health.set(1)
    state.update_atomic_conflicts.inc()
    state.update_atomic_retries.observe(2)

    expected = {
        'twinkle.training.instances.active',
        'twinkle.training.instance.startup.duration_seconds',
        'twinkle.training.instance.shutdown.duration_seconds',
        'twinkle.training.placement_group.allocation.duration_seconds',
        'twinkle.training.placement_group.removal.duration_seconds',
        'twinkle.training.quarantined.total',
        'twinkle.training.sweep.total',
        'twinkle.training.sweep.duration_seconds',
        'twinkle.training.devices.allocated',
        'twinkle.training.placement_groups.active',
        'twinkle.state.redis.health',
        'twinkle.state.redis.update_atomic.conflicts.total',
        'twinkle.state.redis.update_atomic.retries',
    }
    assert expected <= meter.instruments.keys()
    for instrument in meter.instruments.values():
        for _, attributes in instrument.calls:
            assert set(attributes) <= _ALLOWED_ATTRIBUTES
            assert not {'token', 'session_id', 'training_id', 'instance_id', 'request_id'} & set(attributes)


def test_training_metric_adapters_rebind_after_registry_reset(monkeypatch):
    first, _ = build_registry(monkeypatch)
    first_metrics = first.training_metrics()
    metrics.MetricsRegistry.reset()
    second_metrics = metrics.MetricsRegistry.get().training_metrics()
    assert first_metrics is not second_metrics
    assert first_metrics.startup_seconds._instrument is not second_metrics.startup_seconds._instrument


@pytest.mark.parametrize('backend', ['transformers', 'megatron'])
def test_controller_metrics_clear_old_states_and_released_resources(monkeypatch, backend):
    registry, meter = build_registry(monkeypatch)
    spec = SimpleNamespace(model=SimpleNamespace(backend=backend, device_type='cuda'))
    controller = TrainingController.__new__(TrainingController)
    controller.catalog = SimpleNamespace(get=lambda base_model, mode: spec)
    controller._metrics = registry.training_metrics()
    controller._instance_metrics = TrainingInstanceMetrics(controller._metrics)
    controller.instances = {
        str(index): SimpleNamespace(
            instance_id=f'instance_{index}', base_model='private-model', mode='lora',
            state='READY', resident=False, devices=[index * 2, index * 2 + 1],
            application_name=f'private-app-{index}', training_ids=[f'training_{index}'])
        for index in range(2)
    }

    def totals(name):
        values = defaultdict(float)
        for delta, attributes in meter.instruments[name].calls:
            assert set(attributes) <= _ALLOWED_ATTRIBUTES
            assert not {'instance_id', 'base_model', 'application_name', 'training_id'} & set(attributes)
            values[tuple(sorted(attributes.items()))] += delta
        return values

    instances_name = 'twinkle.training.instances.active'
    devices_name = 'twinkle.training.devices.allocated'
    groups_name = 'twinkle.training.placement_groups.active'
    controller._report_instance_metrics()
    first_calls = {name: len(meter.instruments[name].calls)
                   for name in (instances_name, devices_name, groups_name)}
    controller._report_instance_metrics()
    assert first_calls == {name: len(meter.instruments[name].calls) for name in first_calls}
    assert sum(totals(instances_name).values()) == 2
    assert sum(totals(devices_name).values()) == 4
    assert sum(totals(groups_name).values()) == 2

    controller.instances['0'].state = 'DRAINING'
    controller._report_instance_metrics()
    state_totals = {dict(key)['state']: value for key, value in totals(instances_name).items()}
    assert state_totals == {'READY': 1, 'DRAINING': 1}
    assert sum(totals(devices_name).values()) == 4
    assert sum(totals(groups_name).values()) == 2

    for instance in controller.instances.values():
        instance.state = 'STOPPED'
    controller._report_instance_metrics()
    for name in (instances_name, devices_name, groups_name):
        assert all(value == 0 for value in totals(name).values())
