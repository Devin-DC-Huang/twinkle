# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run the full training matrix on an isolated server; runtimes start from zero on client demand."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from xml.etree import ElementTree

import ray
import yaml
from ray import serve

from twinkle.server.config import ServerConfig
from twinkle.server.launcher.server_launcher import ServerLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, choices=['transformers', 'megatron'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    namespace = 'matrix_' + uuid.uuid4().hex[:8]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    source = root / 'tests/server/config' / (
        'server_config_4b_e2e.yaml' if args.backend == 'transformers' else 'server_config_4b_e2e_megatron.yaml')
    raw = yaml.safe_load(source.read_text())
    raw['ray_namespace'] = namespace
    raw['http_options'] = {'host': '127.0.0.1', 'port': port}
    for app in raw['applications']:
        params = app['args']
        if app['import_path'] == 'model':
            params['model_id'] = args.model
            params['nproc_per_node'] = 2
            params['device_group']['ranks'] = 2
            params['device_mesh'] = {'device_type': 'cuda', 'dp_size': 2}
            params['min_replicas'] = 0
            params['max_replicas'] = 2
            params['idle_timeout_s'] = 3600
            params['adapter_config'] = {'adapter_timeout': 3600, 'adapter_max_lifetime': 36000}
            params['env_vars'] = {'TWINKLE_FAIL_FAST': '0', 'TWINKLE_TRUST_REMOTE_CODE': '1'}
            app.pop('deployments', None)
        elif app['import_path'] == 'sampler':
            params['model_id'] = args.model
            params['engine_args']['attention_config'] = {'flash_attn_version': 3}
    config = ServerConfig.model_validate(raw)
    (run_dir / 'server.yaml').write_text(yaml.safe_dump(config.to_yaml_dict(), sort_keys=False))
    os.environ.update({
        'TWINKLE_STATE_RAY_NAMESPACE': namespace,
        'TWINKLE_PERSISTENCE_MODE': 'memory',
        'TWINKLE_DEFAULT_SAVE_DIR': str(run_dir / 'checkpoints'),
        'TWINKLE_MODEL_ID_ALIASES': json.dumps({'Qwen/Qwen3.5-4B': args.model}),
        'TWINKLE_TRUST_REMOTE_CODE': '1',
        'TWINKLE_FAIL_FAST': '0',
        'TWINKLE_TEST_GPU_E2E': '1',
        'TWINKLE_TEST_BACKEND': args.backend,
        'TWINKLE_SERVER_URL': f'http://127.0.0.1:{port}',
        'TWINKLE_TEST_OPERATION_TIMEOUT': '600',
        'PYTHONPATH': str(root / 'src'),
    })
    evidence = {'backend': args.backend, 'namespace': namespace, 'cases': [], 'lazy_start': False}
    ray.init(address='local', namespace=namespace, num_cpus=24, num_gpus=3,
             _temp_dir='/tmp/twm-' + namespace[-8:], include_dashboard=False)
    launcher = ServerLauncher(config)
    try:
        launcher._start_serve()
        for app in config.applications:
            launcher._deploy_application(app)
        launcher._deploy_training()
        assert ray.get(launcher._training_controller.diagnostics.remote())['instances'] == []
        evidence['lazy_start'] = True
        for filename in ['test_sft_e2e.py', 'test_dpo_e2e.py', 'test_grpo_e2e.py', 'test_full_cycle_e2e.py']:
            start = time.monotonic()
            with (run_dir / (filename + '.log')).open('w') as output:
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', str(root / 'tests/server/integration' / filename),
                    '-vv', '-s', '--junitxml=' + str(run_dir / (filename + '.xml')),
                ], stdout=output, stderr=subprocess.STDOUT, env=os.environ.copy(), timeout=7200)
            junit = run_dir / (filename + '.xml')
            cases = ElementTree.parse(junit).getroot().findall('.//testcase') if junit.exists() else []
            skipped = sum(case.find('skipped') is not None for case in cases)
            evidence['cases'].append({'file': filename, 'returncode': result.returncode,
                                      'tests': len(cases), 'skipped': skipped,
                                      'seconds': time.monotonic() - start})
            (run_dir / 'results.json').write_text(json.dumps(evidence, indent=2))
            assert result.returncode == 0, f'{filename} failed; see {run_dir / (filename + ".log")}'
            assert len(cases) == (1 if filename == 'test_full_cycle_e2e.py' else 2) and skipped == 0
    finally:
        try:
            controller = getattr(launcher, '_training_controller', None)
            if controller is not None:
                ray.get(controller.shutdown.remote(), timeout=180)
                diagnostic = ray.get(controller.diagnostics.remote(), timeout=30)
                assert all(row['state'] == 'STOPPED' for row in diagnostic['instances'])
                evidence['training_resources_released'] = True
            launcher._shutdown()
        finally:
            serve.shutdown()
            ray.shutdown()
            (run_dir / 'results.json').write_text(json.dumps(evidence, indent=2))
    evidence['passed'] = True
    (run_dir / 'results.json').write_text(json.dumps(evidence, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
