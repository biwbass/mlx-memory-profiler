"""
In-process memory snapshots for scripts that call MLX directly (mlx-lm,
custom generation loops, etc.). More precise than watching RSS from outside,
since it reads MLX's own allocator counters for the unified-memory pool.

Usage:
    from mlx_mem import mlx_memory_report, track_mlx_memory

    with track_mlx_memory("generate"):
        response = model.generate(prompt, max_tokens=200)

    print(mlx_memory_report())
"""
import contextlib
import time

import mlx.core as mx


def _stat(*names):
    for name in names:
        fn = getattr(mx, name, None) or getattr(getattr(mx, "metal", None), name, None)
        if fn is not None:
            return fn()
    return None


def _active():
    return _stat("get_active_memory")


def _peak():
    return _stat("get_peak_memory")


def _cache():
    return _stat("get_cache_memory")


def mlx_memory_report():
    active, peak, cache = _active(), _peak(), _cache()
    parts = []
    if active is not None:
        parts.append(f"active={active / 1024 ** 3:.2f} GB")
    if cache is not None:
        parts.append(f"cache={cache / 1024 ** 3:.2f} GB")
    if peak is not None:
        parts.append(f"peak={peak / 1024 ** 3:.2f} GB")
    return ", ".join(parts) if parts else "MLX memory API not available in this mlx version"


@contextlib.contextmanager
def track_mlx_memory(label=""):
    before = _active() or 0
    t0 = time.time()
    yield
    dt = time.time() - t0
    after = _active() or 0
    peak = _peak()
    tag = f"[{label}] " if label else ""
    peak_str = f", peak={peak / 1024 ** 3:.2f} GB" if peak is not None else ""
    print(f"{tag}{dt:.2f}s  active {before / 1024 ** 3:.2f} -> {after / 1024 ** 3:.2f} GB{peak_str}")
