import logging
import pytest
from twinkle.server.resources.instances import InstanceLifecycle


@pytest.mark.asyncio
async def test_spawn_logs_exception_with_traceback(caplog, monkeypatch):
    # The ``twinkle`` logger disables propagation, so attach caplog to the lifecycle logger directly.
    target = logging.getLogger('twinkle.server.resources.instances')
    monkeypatch.setattr(target, 'handlers', [*target.handlers, caplog.handler])
    lifecycle = InstanceLifecycle('Training')

    async def broken():
        raise RuntimeError('background sentinel')

    with caplog.at_level(logging.ERROR):
        lifecycle.spawn(broken())
        await lifecycle.join()
    records = [record for record in caplog.records if record.getMessage() == 'Training background job failed']
    assert records and records[0].exc_info[1].args == ('background sentinel',)
    assert not lifecycle.jobs


