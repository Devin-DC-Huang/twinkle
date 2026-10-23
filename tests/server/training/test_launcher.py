# Copyright (c) ModelScope Contributors. All rights reserved.
import socket
import uuid

import pytest
import ray

from twinkle.server.config import ServerConfig
from twinkle.server.launcher.server_launcher import ServerLauncher
from twinkle.server.training.actor import controller_name


def test_failed_initialization_does_not_leave_an_actor_that_blocks_a_retry():
    namespace = 'launcher-retry-' + uuid.uuid4().hex
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    devices = int(ray.cluster_resources()['CPU']) + 1
    payload = {
        'ray_namespace': namespace, 'http_options': {'host': '127.0.0.1', 'port': port},
        'applications': [{'name': 'retry', 'route_prefix': '/api/v1/model/retry', 'import_path': 'model',
                          'args': {'model_id': 'mock', 'backend': 'mock', 'train_mode': 'full',
                                   'device_group': {'ranks': devices, 'device_type': 'cpu'},
                                   'device_mesh': {'dp_size': devices}, 'min_replicas': 1}}],
    }
    launcher = ServerLauncher(ServerConfig.model_validate(payload))
    launcher._start_serve()
    try:
        with pytest.raises(ray.exceptions.RayTaskError, match='No single-node allocation'):
            launcher._deploy_training()
        with pytest.raises(ValueError):
            ray.get_actor(controller_name(namespace), namespace=namespace)
        payload['applications'][0]['args']['min_replicas'] = 0
        launcher.config = ServerConfig.model_validate(payload)
        launcher._deploy_training()
        diagnostic = ray.get(launcher._training_controller.diagnostics.remote())
        assert diagnostic['ready'] and diagnostic['instances'] == []
    finally:
        launcher._shutdown()


def test_cleanup_attempts_all_owners_and_reports_the_original_failure():
    calls = []
    error = ConnectionError('shutdown ACK unknown')

    def failed():
        calls.append('failed')
        raise error

    def healthy():
        calls.append('healthy')

    with pytest.raises(RuntimeError, match='cleanup is incomplete') as caught:
        ServerLauncher._run_cleanup([('first', failed), ('second', healthy)], raise_errors=True)
    assert calls == ['failed', 'healthy']
    assert caught.value.__cause__ is error


def test_startup_rollback_preserves_startup_error_and_cleans_all_owners(monkeypatch):
    config = ServerConfig.model_validate({'applications': [
        {'name': 'gateway', 'import_path': 'server', 'route_prefix': '/api/v1'}]})
    launcher = ServerLauncher(config)
    calls = []
    error = ValueError('invalid application')
    monkeypatch.setattr('twinkle.server.launcher.server_launcher.apply_ray_serve_patches', lambda: None)
    monkeypatch.setattr(launcher, '_configure_environment', lambda: None)
    monkeypatch.setattr(launcher, '_init_ray', lambda: None)
    monkeypatch.setattr(launcher, '_start_serve', lambda: None)

    def deploy(app):
        raise error

    def failed_cleanup():
        calls.append('training')
        raise ConnectionError('cleanup pending')

    monkeypatch.setattr(launcher, '_deploy_application', deploy)
    monkeypatch.setattr(launcher, '_stop_training_controller', failed_cleanup)
    monkeypatch.setattr('twinkle.server.telemetry.flush_telemetry_safely', lambda: calls.append('telemetry'))
    with pytest.raises(ValueError) as caught:
        launcher.launch()
    assert caught.value is error
    assert calls == ['training', 'telemetry']
