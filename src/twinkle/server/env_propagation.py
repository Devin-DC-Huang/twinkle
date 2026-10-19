# Copyright (c) ModelScope Contributors. All rights reserved.
"""Collect environment variables propagated to Ray worker processes."""
from __future__ import annotations

import os

TELEMETRY_ENV_KEYS: tuple[str, ...] = (
    'TWINKLE_STATE_RAY_NAMESPACE',
    'TWINKLE_TELEMETRY_ENABLED',
    'TWINKLE_TELEMETRY_DEBUG',
    'TWINKLE_TELEMETRY_SERVICE',
    'TWINKLE_TELEMETRY_ENDPOINT',
    'TWINKLE_TELEMETRY_INTERVAL',
    'TWINKLE_MODEL_ID_ALIASES',
)


def build_telemetry_env_vars() -> dict[str, str]:
    return {key: os.environ[key] for key in TELEMETRY_ENV_KEYS if key in os.environ}


def build_persistence_env_vars() -> dict[str, str]:
    from twinkle.server.config.persistence import PERSISTENCE_ENV_KEYS
    return {key: os.environ[key] for key in PERSISTENCE_ENV_KEYS if key in os.environ}


def build_server_state_env_vars() -> dict[str, str]:
    from twinkle.server.config.application_spec import SERVER_STATE_ENV_KEYS
    return {key: os.environ[key] for key in SERVER_STATE_ENV_KEYS if key in os.environ}


def build_propagated_env_vars() -> dict[str, str]:
    merged: dict[str, str] = {}
    merged.update(build_telemetry_env_vars())
    merged.update(build_persistence_env_vars())
    merged.update(build_server_state_env_vars())
    return merged
