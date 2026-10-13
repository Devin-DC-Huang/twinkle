# Copyright (c) ModelScope Contributors. All rights reserved.
"""Safe-mode checks at the remote dispatch boundary, using real Ray actors."""
import pytest
import ray
import os

import twinkle.infra as infra
from twinkle import remote_function


class _Component:
    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def run(self, value):
        assert infra._mode == 'ray'
        assert os.environ['TWINKLE_TRUST_REMOTE_CODE'] == '0'
        return self.apply(value, transform=lambda x: x + 1)

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def apply(self, value, transform):
        return transform(value)


@pytest.fixture(scope='module')
def actor():
    ray.init(num_cpus=1, include_dashboard=False, logging_level='ERROR')
    try:
        yield ray.remote(_Component).options(
            runtime_env={'env_vars': {'TWINKLE_MODE': 'ray', 'TWINKLE_TRUST_REMOTE_CODE': '0'}}).remote()
    finally:
        ray.shutdown()


@pytest.fixture
def handle(actor, monkeypatch):
    monkeypatch.setattr(infra, '_mode', 'ray')
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '0')
    instance = _Component()
    instance._actors = [actor]
    return instance


@pytest.mark.parametrize('trust', ['0', '1'])
def test_worker_can_call_another_method_with_local_callable(handle, monkeypatch, trust):
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', trust)
    assert handle.run(3) == 4


@pytest.mark.parametrize('value', [lambda: None, type, {'nested': [lambda: None]}])
def test_remote_handle_rejects_callable_inputs(handle, value):
    with pytest.raises(ValueError, match='Callable or Type'):
        handle.apply(3, transform=value)
