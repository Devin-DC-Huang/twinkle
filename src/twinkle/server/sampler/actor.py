# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ray composition for the sampling controller and its owned driver."""
import asyncio

from twinkle.server.config.application_spec import SamplerArgs
from .controller import SamplerController


def create_controller_actor():
    import ray

    from .driver import SamplerDriver

    @ray.remote(num_cpus=0.1, max_restarts=-1, max_task_retries=0)
    class SamplerControllerActor(SamplerController):

        def __init__(self, values, namespace, pool_id, options, budgets):
            args = SamplerArgs.model_validate(values)
            super().__init__(args, SamplerDriver(args, namespace, pool_id, options, budgets), budgets)

        async def initialize(self):
            await super().initialize()
            if not self._stopping and self._monitor_task is None:
                self._monitor_task = asyncio.create_task(self._monitor())

        async def shutdown(self):
            await super().shutdown()
            await self.driver.runtime.wait_for_cleanup()

    return SamplerControllerActor
