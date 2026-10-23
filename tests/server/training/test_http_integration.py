import json
import subprocess
import sys
import uuid
from pathlib import Path


def test_installed_native_and_tinker_sdks_use_managed_http_routes():
    # This probe owns a separate Ray cluster. Stop the session fixture's cluster
    # first so two local GCS/raylet pairs do not evict each other on a slow host.
    import ray
    ray.shutdown()
    try:
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).resolve().parents[3]
                / 'tests/server/integration/on_demand_training/http_probe.py')
        ], capture_output=True, text=True, timeout=1800)
    finally:
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            log_to_driver=False,
            namespace='twinkle_test',
            _temp_dir='/tmp/twinkle-test-' + uuid.uuid4().hex[:8])
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = next(json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"passed":'))
    assert evidence['passed'] and evidence['native_recreation'] and evidence['tinker_shared']
    assert len(set(evidence['native_stable_routing'].values())) == 2
    assert len(set(evidence['tinker_stable_routing'].values())) == 2
    assert evidence['native_lora']
    assert evidence['runtime_lifecycle_rejected']
    assert evidence['tinker_full']
    assert evidence['restart'] and evidence['state_loss'] and evidence['cycles'] == 10
