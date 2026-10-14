# Copyright (c) ModelScope Contributors. All rights reserved.
"""Explicit real-device acceptance; run this file only after local mock checks.

Required: TWINKLE_DEVICE_BACKEND, TWINKLE_DEVICE_MODEL, TWINKLE_DEVICE_OTHER_MODEL,
TWINKLE_DEVICE_RUN_DIR and four exclusively available visible devices.
This is intentionally outside pytest's default test filename pattern.
"""
import json
import numpy as np
import os
import pytest
import ray
import requests
import socket
import time
import uuid
from on_demand_sdk import unload_tinker_model
from pathlib import Path
from ray import serve

from twinkle.server.config import ServerConfig
from twinkle.server.launcher.server_launcher import ServerLauncher
from twinkle_client import init_tinker_client
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import ClientContext
from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel


@pytest.fixture(scope='module')
def server():
    backend = os.environ['TWINKLE_DEVICE_BACKEND']
    assert backend in ('transformers', 'megatron')
    run_dir = Path(os.environ['TWINKLE_DEVICE_RUN_DIR'])
    run_dir.mkdir(parents=True, exist_ok=True)
    namespace = 'device_' + uuid.uuid4().hex[:8]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    os.environ['TWINKLE_STATE_RAY_NAMESPACE'] = namespace
    os.environ['TWINKLE_DEFAULT_SAVE_DIR'] = str(run_dir / 'checkpoints')
    os.environ['TWINKLE_PERSISTENCE_MODE'] = 'memory'
    os.environ['TWINKLE_LONG_POLL_TIMEOUT'] = '2'
    mesh = {'fsdp_size': 2} if backend == 'transformers' else {'tp_size': 2}
    apps = []
    for alias, variable in [('small', 'TWINKLE_DEVICE_MODEL'), ('other', 'TWINKLE_DEVICE_OTHER_MODEL')]:
        source = os.environ[variable]
        assert Path(source, 'config.json').is_file()
        mode = {'num_devices': 2, 'device_mesh': mesh, 'idle_timeout_s': 600 if alias == 'small' else 30}
        if backend == 'transformers':
            mode['strategy'] = 'native_fsdp'
        apps.append({
            'name': alias,
            'route_prefix': '/api/v1/model/' + alias,
            'import_path': 'model',
            'args': {
                'model_id': source,
                'backend': backend,
                'device_type': 'cuda',
                'modes': {
                    'lora': {
                        **mode, 'max_loras': 2,
                        'max_r': 8
                    },
                    'full': mode
                },
            }
        })
    apps.append({
        'name': 'gateway',
        'route_prefix': '/api/v1',
        'import_path': 'server',
        'args': {
            'server_config': {
                'per_token_model_limit': 10
            }
        },
        'deployments': [{
            'ray_actor_options': {
                'num_cpus': 0.1
            }
        }]
    })
    os.environ['TWINKLE_MODEL_ID_ALIASES'] = json.dumps(
        {app['name']: app['args']['model_id']
         for app in apps if app['import_path'] == 'model'})
    config = ServerConfig(ray_namespace=namespace, http_options={'host': '127.0.0.1', 'port': port}, applications=apps)
    import yaml
    (run_dir / 'server.yaml').write_text(yaml.safe_dump(config.to_yaml_dict(), sort_keys=False))
    ray.init(
        address='local',
        namespace=namespace,
        num_cpus=16,
        num_gpus=4,
        _temp_dir='/tmp/twd-' + namespace[-8:],
        include_dashboard=False,
        log_to_driver=True)
    launcher = ServerLauncher(config)
    launcher._start_serve()
    evidence = {
        'backend': backend,
        'namespace': namespace,
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cases': []
    }
    try:
        for app in config.applications:
            launcher._deploy_application(app)
        launcher._deploy_training()
        controller = launcher._training_controller
        base = f'http://127.0.0.1:{port}'
        assert ray.get(controller.diagnostics.remote())['instances'] == []
        evidence['ray_address'] = ray.get_runtime_context().gcs_address
        yield base, controller, evidence, launcher
    finally:
        if getattr(launcher, '_training_controller', None) is not None:
            ray.get(launcher._training_controller.shutdown.remote(), timeout=180)
            ray.kill(launcher._training_controller, no_restart=True)
        (run_dir / 'results.json').write_text(json.dumps(evidence, indent=2))
        serve.shutdown()
        ray.shutdown()


class Handle:

    def __init__(self, base, protocol, mode, alias='small'):
        self.base, self.protocol = base, protocol
        self.token = 'device-validation-owner'
        session = requests.post(base + '/api/v1/twinkle/create_session', json={}, timeout=20).json()['session_id']
        self.transport = ClientTransport(ClientContext(base_url=base, api_key=self.token, session_id=session))
        self.rate = 1e-3 if mode == 'lora' else 5e-5
        if protocol == 'native':
            from peft import LoraConfig
            self.client = MultiLoraTransformersModel(alias, transport=self.transport)
            self.client.add_adapter_to_model('same',
                                             LoraConfig(r=8, target_modules='all-linear') if mode == 'lora' else None)
            self.identity = self.client._training_ids['same']
            self.client.set_loss('CrossEntropyLoss')
            self.client.set_optimizer('Adam', lr=self.rate)
            self.client.set_processor('InputProcessor', padding_side='right')
        else:
            init_tinker_client()
            from tinker import ServiceClient
            self.service = ServiceClient(base_url=base, api_key=self.token)
            self.client = (
                self.service.create_lora_training_client(base_model=alias, rank=8)
                if mode == 'lora' else self.service.create_full_training_client(base_model=alias))
            self.identity = self.client.model_id
        # Fixed token IDs need no external dataset, and every position has a target.
        self.inputs = [{
            'input_ids': [100, 200, 300, 400, 500, 600, 700, 800],
            'labels': [200, 300, 400, 500, 600, 700, 800, 900],
            'position_ids': list(range(8)),
            'attention_mask': [1] * 8
        }] * 2

    def datum(self):
        from tinker import types
        feature = self.inputs[0]
        return [
            types.Datum(
                model_input=types.ModelInput.from_ints(feature['input_ids']),
                loss_fn_inputs={
                    'target_tokens': types.TensorData(data=feature['labels'], dtype='int64', shape=[8]),
                    'weights': types.TensorData(data=[1.] * 8, dtype='float32', shape=[8]),
                }) for _ in self.inputs
        ]

    def evaluate(self):
        if self.protocol == 'native':
            result = self.client.forward_only(self.inputs).result
            values = result['logps']
        else:
            result = self.client.forward(self.datum(), 'cross_entropy').result()
            values = result.loss_fn_outputs[0]['logprobs'].tolist()
        values = np.asarray(values, dtype=float).reshape(-1)
        assert values.size and np.isfinite(values).all(), values
        return values

    def step(self, *, contrast=False):
        labels = self.inputs[0]['labels']
        if contrast:
            self.inputs[0]['labels'] = list(reversed(labels))
        try:
            if self.protocol == 'native':
                self.client.forward_backward(self.inputs)
                self.client.clip_grad_and_step()
            else:
                from tinker import types
                self.client.forward_backward(self.datum(), 'cross_entropy').result()
                self.client.optim_step(types.AdamParams(learning_rate=self.rate)).result()
        finally:
            self.inputs[0]['labels'] = labels

    def save(self):
        name = 'device-' + uuid.uuid4().hex[:8]
        return (self.client.save(name, save_optimizer=True).twinkle_path
                if self.protocol == 'native' else self.client.save_state(name).result().path)

    def restore(self, path):
        if self.protocol == 'native':
            self.client.resume_from_checkpoint(path)
        else:
            self.client.load_state_with_optimizer(path).result()

    def restore_weights(self, path):
        if self.protocol == 'native':
            self.client.resume_from_checkpoint(path, resume_only_model=True)
        else:
            self.client.load_state(path).result()

    def close(self):
        if self.protocol == 'native':
            self.client.remove_adapter()
        else:
            assert unload_tinker_model(self.service, self.identity).model_id == self.identity
        self.transport.close()


def active_instances(controller):
    return [row for row in ray.get(controller.diagnostics.remote())['instances'] if row['state'] != 'STOPPED']


@pytest.mark.parametrize('protocol', ['native', 'tinker'])
@pytest.mark.parametrize('mode', ['lora', 'full'])
def test_training_isolation_checkpoint_and_device_reclamation(server, protocol, mode):
    base, controller, evidence, _ = server
    first, second = Handle(base, protocol, mode), Handle(base, protocol, mode)
    assert first.identity != second.identity
    instances = active_instances(controller)
    assert len(instances) == (1 if mode == 'lora' else 2)
    assert all(len(row['devices']) == 2 for row in instances)
    devices = [device for row in instances for device in row['devices']]
    assert len(set(devices)) == len(devices)
    original, untouched = first.evaluate(), second.evaluate()
    untouched_checkpoint = second.save() if mode == 'lora' else None
    for _ in range(4):
        first.step()
    trained = first.evaluate()
    assert float(trained.mean()) > float(original.mean()) + 1e-4, (original, trained)
    np.testing.assert_allclose(second.evaluate(), untouched, rtol=0, atol=2e-3)
    if mode == 'lora':
        third, fourth = Handle(base, protocol, mode), Handle(base, protocol, mode)
        assert len(active_instances(controller)) == 2
        assert ray.available_resources().get('GPU', 0) == 0
        with pytest.raises(Exception, match='(?i)allocation'):
            Handle(base, protocol, mode)
        fourth.close()
        third.restore(untouched_checkpoint)
        second.step()
        third.step()
        np.testing.assert_allclose(second.evaluate(), third.evaluate(), rtol=0, atol=2e-2)
        np.testing.assert_allclose(first.evaluate(), trained, rtol=0, atol=2e-3)
        third.close()
        assert len(active_instances(controller)) == 1
    checkpoint = first.save()
    first.step(contrast=True)
    continuation = first.evaluate()
    first.restore(checkpoint)
    np.testing.assert_allclose(first.evaluate(), trained, rtol=0, atol=2e-3)
    first.step(contrast=True)
    np.testing.assert_allclose(first.evaluate(), continuation, rtol=0, atol=2e-2)
    old_id = first.identity
    first.close()
    assert len(active_instances(controller)) == 1
    replacement = Handle(base, protocol, mode)
    assert replacement.identity != old_id
    replacement.restore_weights(checkpoint)
    np.testing.assert_allclose(replacement.evaluate(), trained, rtol=0, atol=2e-3)
    replacement.step(contrast=True)
    optimizer_reset = replacement.evaluate()
    assert float(np.max(np.abs(optimizer_reset - continuation))) > 2e-2, 'Optimizer continuation oracle is not decisive'
    replacement.restore(checkpoint)
    np.testing.assert_allclose(replacement.evaluate(), trained, rtol=0, atol=2e-3)
    replacement.step(contrast=True)
    np.testing.assert_allclose(replacement.evaluate(), continuation, rtol=0, atol=2e-2)
    replacement.close()
    second.close()
    assert not active_instances(controller)
    assert ray.available_resources().get('GPU', 0) == 4
    evidence['cases'].append({
        'protocol': protocol,
        'mode': mode,
        'initial_logps': original.tolist(),
        'trained_logps': trained.tolist(),
        'continuation_logps': continuation.tolist(),
        'optimizer_reset_logps': optimizer_reset.tolist(),
        'checkpoint': checkpoint,
        'devices': devices,
        'passed': True
    })


def test_cross_model_reuse_and_heartbeat_does_not_prevent_expiry(server):
    base, controller, evidence, _ = server
    before = ray.available_resources().get('GPU', 0)
    for index in range(10):
        handle = Handle(base, 'native', 'full', alias='small' if index % 2 == 0 else 'other')
        values = handle.evaluate()
        handle.close()
        assert not active_instances(controller)
        assert ray.available_resources().get('GPU', 0) == before
        print(json.dumps({'cycle': index + 1, 'logps': values.tolist()}), flush=True)
    expired = Handle(base, 'native', 'lora', alias='other')
    expired.evaluate()
    deadline = time.monotonic() + 60
    while active_instances(controller) and time.monotonic() < deadline:
        response = expired.transport.post(
            base + '/api/v1/twinkle/session_heartbeat', json_data={'session_id': expired.transport.context.session_id})
        response.raise_for_status()
        time.sleep(2)
    assert not active_instances(controller), 'Heartbeats incorrectly kept idle training alive'
    with pytest.raises(Exception):
        expired.client.zero_grad()
    assert ray.available_resources().get('GPU', 0) == before
    expired.transport.close()
    evidence['cases'].append({'cycles': 10, 'idle_reclaimed': True, 'passed': True})


@pytest.mark.parametrize('protocol', ['native', 'tinker'])
def test_idle_lora_user_expires_without_resetting_active_user(server, protocol):
    base, controller, evidence, _ = server
    idle = Handle(base, protocol, 'lora', alias='other')
    active = Handle(base, protocol, 'lora', alias='other')
    instance = active_instances(controller)[0]
    assert instance['training_ids'] == [idle.identity, active.identity]
    original = idle.evaluate()
    for _ in range(4):
        active.step()
    trained = active.evaluate()
    assert trained.mean() > original.mean() + 1e-4
    np.testing.assert_allclose(idle.evaluate(), original, rtol=0, atol=2e-3)
    checkpoint = active.save()
    active.step(contrast=True)
    expected = active.evaluate()
    active.restore(checkpoint)
    deadline = time.monotonic() + 60
    while True:
        np.testing.assert_allclose(active.evaluate(), trained, rtol=0, atol=2e-3)
        response = idle.transport.post(
            base + '/api/v1/twinkle/session_heartbeat', json_data={'session_id': idle.transport.context.session_id})
        response.raise_for_status()
        instances = active_instances(controller)
        assert len(instances) == 1 and instances[0]['instance_id'] == instance['instance_id']
        if idle.identity not in instances[0]['training_ids']:
            break
        assert time.monotonic() < deadline, 'Idle LoRA user did not expire while the other user stayed active'
        time.sleep(2)
    with pytest.raises(Exception):
        idle.evaluate()
    active.step(contrast=True)
    np.testing.assert_allclose(active.evaluate(), expected, rtol=0, atol=2e-2)
    replacement = Handle(base, protocol, 'lora', alias='other')
    assert replacement.identity != idle.identity
    assert active_instances(controller)[0]['instance_id'] == instance['instance_id']
    np.testing.assert_allclose(replacement.evaluate(), original, rtol=0, atol=2e-3)
    replacement.close()
    np.testing.assert_allclose(active.evaluate(), expected, rtol=0, atol=2e-3)
    active.close()
    idle.transport.close()
    assert not active_instances(controller)
    assert ray.available_resources().get('GPU', 0) == 4
    evidence['cases'].append({
        'protocol': protocol,
        'idle_lora_expired': True,
        'active_optimizer_preserved': True,
        'slot_reused': True,
        'passed': True
    })


def test_resident_full_reset_preserves_other_instances(server):
    base, controller, evidence, launcher = server
    raw = launcher.config.to_yaml_dict()
    raw['applications'][0]['args']['modes']['full']['min_replicas'] = 1
    ray.get(controller.shutdown.remote(), timeout=180)
    ray.kill(controller, no_restart=True)
    from twinkle.server.training.actor import controller_name
    deadline = time.monotonic() + 30
    while True:
        try:
            ray.get_actor(controller_name(launcher.config.ray_namespace), namespace=launcher.config.ray_namespace)
        except ValueError:
            break
        assert time.monotonic() < deadline, 'Previous controller name was not released'
        time.sleep(0.1)
    replacement = ServerLauncher(ServerConfig.model_validate(raw))
    replacement._deploy_training()
    controller = replacement._training_controller
    launcher._training_controller = controller
    evidence['resident_controller_epoch'] = ray.get(controller.diagnostics.remote())['epoch']
    resident, extra = Handle(base, 'native', 'full'), Handle(base, 'native', 'full')
    instances = active_instances(controller)
    resident_instance = next(row for row in instances if row['resident'])
    original, untouched = resident.evaluate(), extra.evaluate()
    for _ in range(2):
        resident.step()
    assert resident.evaluate().mean() > original.mean() + 1e-4
    resident.close()
    restored = Handle(base, 'native', 'full')
    instances = active_instances(controller)
    assert next(row for row in instances if row['resident'])['instance_id'] == resident_instance['instance_id']
    np.testing.assert_allclose(restored.evaluate(), original, rtol=0, atol=2e-3)
    np.testing.assert_allclose(extra.evaluate(), untouched, rtol=0, atol=2e-3)
    restored.close()
    extra.close()
    instances = active_instances(controller)
    assert len(instances) == 1 and instances[0]['resident']
    assert ray.available_resources().get('GPU', 0) == 2
    evidence['cases'].append({'resident_reset': True, 'extra_preserved': True, 'passed': True})


def test_worker_failure_reclaims_devices_and_preserves_resident(server):
    from ray.util import list_named_actors
    base, _, evidence, launcher = server
    controller = launcher._training_controller
    resident = Handle(base, 'native', 'full')
    original = resident.evaluate()
    failing = Handle(base, 'native', 'lora')
    failing.evaluate()
    instance = next(row for row in active_instances(controller) if row['mode'] == 'lora')
    actors = [
        row for row in list_named_actors(all_namespaces=True) if row['namespace'] == evidence['namespace']
        and instance['instance_id'] in row['name'] and not row['name'].endswith('-registry')
    ]
    assert len(actors) == 2, actors
    ray.kill(ray.get_actor(actors[0]['name'], namespace=evidence['namespace']), no_restart=True)
    deadline = time.monotonic() + 90
    while any(row['instance_id'] == instance['instance_id'] for row in active_instances(controller)):
        assert time.monotonic() < deadline, 'Failed runtime devices were not reclaimed'
        time.sleep(1)
    with pytest.raises(Exception):
        failing.client.zero_grad()
    np.testing.assert_allclose(resident.evaluate(), original, rtol=0, atol=2e-3)
    recovered = Handle(base, 'native', 'lora')
    recovered.evaluate()
    recovered.close()
    resident.close()
    failing.transport.close()
    assert ray.available_resources().get('GPU', 0) == 2
    evidence['cases'].append({'worker_failure': True, 'resident_preserved': True, 'recreated': True, 'passed': True})
