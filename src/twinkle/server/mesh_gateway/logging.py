# Copyright (c) ModelScope Contributors. All rights reserved.
"""Keep expired-session heartbeat access logs bounded without hiding other errors."""
from __future__ import annotations

import logging
import time

from .proxy import HEARTBEAT_PATHS


class HeartbeatAccessLogFilter(logging.Filter):
    """Retain the first heartbeat 404, then one summary per minute."""

    def __init__(self, interval_seconds: float = 60.0):
        super().__init__()
        self._interval_seconds = interval_seconds
        self._next_report = 0.0
        self._suppressed = 0

    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn access records contain (client, method, path, HTTP version, status).
        if not isinstance(record.args, tuple) or len(record.args) != 5:
            return True
        _, method, path, _, status = record.args
        if method != 'POST' or path.split('?', 1)[0] not in HEARTBEAT_PATHS or status != 404:
            return True

        now = time.monotonic()
        if now < self._next_report:
            self._suppressed += 1
            return False
        self._next_report = now + self._interval_seconds
        if self._suppressed:
            record.msg += f' (suppressed {self._suppressed} similar heartbeat 404 access logs)'
            self._suppressed = 0
        return True
