# On-demand training validation

The review fixes passed local validation with CPU models, real Ray scheduling,
the Native/Tinker SDKs over HTTP, and isolated memory/Redis state:

- Server, model, Transformers, and client CPU regression: 807 passed, no failures,
  and 21 hardware-specific tests skipped.
- HTTP mock acceptance: passed for both SDKs in shared LoRA and exclusive full
  modes, including release/recreation, runtime lifecycle rejection, controller
  restart, state loss, and ten reuse cycles.
- Existing YAML startup: passed with memory state and with Redis state.
- The separate backend prerequisite (#315): 17 focused CPU regressions passed
  without failures or skips in an isolated checkout.

Repository pre-commit checks passed. Raw local outputs are retained outside the
repository.

Real-device validation of the revised code is deferred at the user's request
because the PPU server is running other tests. PR #312 remains a draft.

The previous runtime (`1bcf657e1534f6023b5de605b5a204c94705729b`) passed nine
acceptance cases on each of Transformers and Megatron, with no failures or skips.
Those results do not validate the revised lifecycle code. Raw logs, JUnit files,
server configurations, and numerical results are retained outside the repository.

Standalone validation programs live in `tools/validation/on_demand_training/`.
The collected CPU tests under `tests/server/training/` run their allocation and
HTTP programs in isolated processes. Device validation is explicitly invoked:

```bash
python -m pytest tools/validation/on_demand_training/device_validation.py -vv
```

It requires the backend, model paths, a separate evidence directory, and free
devices configured by `TWINKLE_DEVICE_*`. It must not run against occupied devices.
