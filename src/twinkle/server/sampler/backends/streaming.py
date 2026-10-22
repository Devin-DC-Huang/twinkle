# Copyright (c) ModelScope Contributors. All rights reserved.
"""Cross-process streaming bridge for sampler backends.

Lives in its own module (rather than the ``backends`` package ``__init__``) so
that a backend implementation (e.g. ``mock_sampler``) can import it without
importing its own package root — which would be a package-initialisation-order
dependency. ``backends/__init__`` re-exports these names, so external consumers
(``from twinkle.server.sampler.backends import stream_to_queue``) are unchanged.
"""
from __future__ import annotations

STREAM_SENTINEL = '__STREAM_END__'


def stream_to_queue(sampler, queue, inputs, sampling_params=None, adapter_name='', adapter_path=None):
    """Push streaming deltas from *sampler* to a cross-process Ray queue.

    Works with any object that exposes a ``sample_stream`` iterator.
    """
    generation = sampler.sample_stream(inputs, sampling_params, adapter_name, adapter_path)
    try:
        for delta, reason in generation:
            queue.put((delta, reason))
    except Exception as e:
        try:
            queue.put(e)
        except Exception:
            pass  # The HTTP consumer may already have closed its queue.
    finally:
        close = getattr(generation, 'close', None)
        if close is not None:
            close()
        try:
            queue.put(STREAM_SENTINEL)
        except Exception:
            pass
